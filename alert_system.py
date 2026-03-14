"""
TRINETRA — Early Warning Alert Dispatch System
Multi-channel alerts: SMS (Twilio), IVR, Email, Webhook, Dashboard push.
Aligned with NDMA (National Disaster Management Authority) protocols.
"""

import asyncio
import aiohttp
import aiofiles
import json
import logging
import smtplib
import os
from datetime import datetime
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict
from twilio.rest import Client as TwilioClient
from pathlib import Path

logger = logging.getLogger('trinetra.alerts')

# ── District Emergency Contacts ───────────────────────────────────────────────
DISTRICT_CONTACTS: Dict[str, dict] = {
    'chamoli': {
        'dm_phone':         '+911362242209',
        'ndrf_phone':       '+919557844829',
        'sdrf_phone':       '+911352712006',
        'sdma_phone':       '+911352712001',
        'email':            ['dm.chamoli@gov.in', 'ndrf.uk@gov.in', 'sdma.uk@gov.in'],
        'population':        412920,
        'vulnerable_villages': ['Joshimath', 'Tapovan', 'Rini', 'Sumna'],
        'relief_camps':     ['IC Joshimath', 'IC Chamoli', 'IC Karnaprayag'],
    },
    'rudraprayag': {
        'dm_phone':         '+911364233522',
        'ndrf_phone':       '+919557844829',
        'sdrf_phone':       '+911352712006',
        'sdma_phone':       '+911352712001',
        'email':            ['dm.rudraprayag@gov.in', 'sdma.uk@gov.in'],
        'population':        242285,
        'vulnerable_villages': ['Kedarnath', 'Triyuginarayan', 'Ukhimath'],
        'relief_camps':     ['IC Rudraprayag', 'IC Agastmuni'],
    },
    'pithoragarh': {
        'dm_phone':         '+919412086208',
        'ndrf_phone':       '+919557844829',
        'sdrf_phone':       '+911352712006',
        'sdma_phone':       '+911352712001',
        'email':            ['dm.pithoragarh@gov.in', 'sdma.uk@gov.in'],
        'population':        483439,
        'vulnerable_villages': ['Munsiyari', 'Dharchula', 'Didihat'],
        'relief_camps':     ['IC Pithoragarh', 'IC Dharchula'],
    },
    'uttarkashi': {
        'dm_phone':         '+911374222201',
        'ndrf_phone':       '+919557844829',
        'sdrf_phone':       '+911352712006',
        'sdma_phone':       '+911352712001',
        'email':            ['dm.uttarkashi@gov.in', 'sdma.uk@gov.in'],
        'population':        330086,
        'vulnerable_villages': ['Gangotri', 'Harshil', 'Bhatwari'],
        'relief_camps':     ['IC Uttarkashi', 'IC Bhatwari'],
    },
    'bageshwar': {
        'dm_phone':         '+919412022228',
        'ndrf_phone':       '+919557844829',
        'sdrf_phone':       '+911352712006',
        'sdma_phone':       '+911352712001',
        'email':            ['dm.bageshwar@gov.in'],
        'population':        259898,
        'vulnerable_villages': ['Baijnath', 'Kapkot', 'Kafligair'],
        'relief_camps':     ['IC Bageshwar', 'IC Baijnath'],
    },
}

# ── Alert Message Templates ───────────────────────────────────────────────────
ALERT_TEMPLATES = {
    'fire': {
        'L2': ('⚠️ FIRE WATCH — {district}. Predicted forest fire risk: {prob:.0%}. '
               'Forest department on standby. Monitor MODIS active fire detections.'),
        'L3': ('⚠️ FOREST FIRE WARNING — {district}. AI risk score: {prob:.0%}. '
               'Deploy fire response teams immediately. Issue community advisory. '
               'Contact: {dm_phone}'),
        'L4': ('🔴 CRITICAL FIRE ALERT — {district}. Risk: {prob:.0%}. '
               'IMMEDIATE evacuation required. All available resources deployed. '
               'Roads may be impassable. Contact DM: {dm_phone}'),
    },
    'flood': {
        'L2': ('⚠️ FLOOD WATCH — {district}. Extreme rainfall predicted: {prob:.0%} risk. '
               'Alert communities along river corridors. '
               'Pre-position boats and rescue equipment.'),
        'L3': ('⚠️ FLOOD WARNING — {district}. Flash flood probability: {prob:.0%}. '
               'EVACUATE low-lying river banks immediately. Open relief camps: {camps}. '
               'Contact: {dm_phone}'),
        'L4': ('🔴 CRITICAL FLOOD ALERT — {district}. Risk: {prob:.0%}. '
               'FLASH FLOOD IMMINENT. Forced evacuation in progress. '
               'Do NOT attempt to cross flooded roads. '
               'Emergency helpline: 1070. DM: {dm_phone}'),
    },
    'landslide': {
        'L2': ('⚠️ LANDSLIDE WATCH — {district}. Slope instability risk: {prob:.0%}. '
               'Alert hill communities. Inspect vulnerable slopes.'),
        'L3': ('⚠️ LANDSLIDE WARNING — {district}. Risk: {prob:.0%}. '
               'Close mountain highways: NH-58, NH-7. '
               'Alert villages on steep slopes. DM: {dm_phone}'),
        'L4': ('🔴 CRITICAL LANDSLIDE — {district}. Risk: {prob:.0%}. '
               'IMMEDIATE EVACUATION of all slope-face settlements. '
               'All mountain roads CLOSED. Emergency: 1070. DM: {dm_phone}'),
    },
}

# IVR audio text (synthesized to call)
IVR_TEMPLATES = {
    'fire': {
        'L3': ('यह एक आपातकालीन संदेश है। {district} जिले में जंगल की आग का खतरा है। '
               'कृपया सुरक्षित स्थान पर जाएं।'),
        'L4': ('यह तत्काल आपातकालीन चेतावनी है। {district} में जंगल की आग की '
               'आपातकालीन स्थिति है। अभी घर छोड़ें।'),
    },
    'flood': {
        'L3': ('यह आपातकालीन संदेश है। {district} में बाढ़ का खतरा है। '
               'नदी किनारे के लोग तुरंत सुरक्षित स्थान जाएं।'),
        'L4': ('तत्काल चेतावनी। {district} में अचानक बाढ़ आ सकती है। '
               'सभी लोग अभी ऊंचे स्थान पर जाएं।'),
    },
    'landslide': {
        'L3': ('आपातकालीन चेतावनी। {district} में भूस्खलन का खतरा है। '
               'पहाड़ी इलाकों के लोग सुरक्षित स्थान जाएं।'),
        'L4': ('तत्काल आपात स्थिति। {district} में भूस्खलन हो सकता है। '
               'सभी पहाड़ी क्षेत्र खाली करें।'),
    },
}


# ── Main AlertDispatcher ──────────────────────────────────────────────────────
class AlertDispatcher:
    """
    Dispatches emergency alerts via:
      1. SMS     — Twilio to DM, NDRF, SDRF, SDMA
      2. IVR     — Twilio voice call with Hindi message
      3. Email   — SMTP to district officials
      4. Webhook — Push to SDMA dashboard, NDMA portal
      5. Log     — Persistent JSONL audit trail
    """

    def __init__(self):
        self.twilio_sid  = os.environ.get('TWILIO_ACCOUNT_SID', '')
        self.twilio_auth = os.environ.get('TWILIO_AUTH_TOKEN',  '')
        self.from_phone  = os.environ.get('TWILIO_PHONE',       '+1xxxxxxxxxx')
        self.webhook_urls = json.loads(os.environ.get('ALERT_WEBHOOKS', '[]'))
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port   = int(os.environ.get('SMTP_PORT', '587'))
        self.smtp_user   = os.environ.get('SMTP_USER', '')
        self.smtp_pass   = os.environ.get('SMTP_PASS', '')
        self.log_dir     = Path('./logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.twilio_sid and self.twilio_auth:
            self.twilio = TwilioClient(self.twilio_sid, self.twilio_auth)
        else:
            self.twilio = None
            logger.warning("[Alerts] Twilio credentials not set — SMS/IVR disabled")

    def _format_message(self, hazard_type: str, level: str,
                         district: str, probability: float,
                         contacts: dict) -> str:
        """Format alert message with district-specific details."""
        template = ALERT_TEMPLATES.get(hazard_type, {}).get(level, '')
        if not template:
            return f"HAZARD ALERT — {district}: {hazard_type.upper()} {level}"
        return template.format(
            district  = district.title(),
            prob      = probability,
            dm_phone  = contacts.get('dm_phone', 'N/A'),
            camps     = ', '.join(contacts.get('relief_camps', [])[:2])
        )

    async def dispatch(self, district: str, hazard_type: str,
                       level: str, probability: float):
        """
        Dispatch alert via all channels concurrently.
        All channels run in parallel for minimum latency.
        """
        district  = district.lower()
        contacts  = DISTRICT_CONTACTS.get(district, {})
        if not contacts:
            logger.warning(f"[Alerts] No contacts for district: {district}")

        sms_msg = self._format_message(hazard_type, level, district,
                                        probability, contacts)
        ivr_msg = IVR_TEMPLATES.get(hazard_type, {}).get(
            level, f'Emergency alert for {district}')
        ivr_msg = ivr_msg.format(district=district)

        tasks = [
            self._send_sms(contacts.get('dm_phone'),   sms_msg, 'DM'),
            self._send_sms(contacts.get('ndrf_phone'), sms_msg, 'NDRF'),
            self._send_sms(contacts.get('sdrf_phone'), sms_msg, 'SDRF'),
            self._send_sms(contacts.get('sdma_phone'), sms_msg, 'SDMA'),
            self._send_ivr(contacts.get('dm_phone'),   ivr_msg),
            self._send_email(
                contacts.get('email', []), sms_msg,
                district, hazard_type, probability, level
            ),
            self._send_webhooks(district, hazard_type, level, probability, contacts),
            self._log_alert(district, hazard_type, level, probability, sms_msg),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = sum(1 for r in results if r is True)
        total   = len([t for t in tasks if t is not None])
        logger.info(
            f"[Alerts] {district}/{hazard_type}/{level} — "
            f"{success}/{total} channels successful"
        )

    async def _send_sms(self, phone: Optional[str],
                         message: str, recipient: str = '') -> bool:
        """Send SMS via Twilio REST API."""
        if not phone or not self.twilio:
            return False
        try:
            msg = self.twilio.messages.create(
                body  = message[:1600],
                from_ = self.from_phone,
                to    = phone
            )
            logger.info(f"  [SMS] {recipient} ({phone}): {msg.sid}")
            return True
        except Exception as e:
            logger.error(f"  [SMS] Failed → {recipient} ({phone}): {e}")
            return False

    async def _send_ivr(self, phone: Optional[str], message: str) -> bool:
        """
        Trigger IVR voice call via Twilio TwiML.
        Uses <Say> verb to synthesise Hindi/English message.
        """
        if not phone or not self.twilio:
            return False
        try:
            twiml = (
                f'<Response>'
                f'<Say language="hi-IN" voice="Polly.Aditi">{message}</Say>'
                f'<Pause length="2"/>'
                f'<Say language="hi-IN" voice="Polly.Aditi">{message}</Say>'
                f'</Response>'
            )
            call = self.twilio.calls.create(
                twiml = twiml,
                from_ = self.from_phone,
                to    = phone
            )
            logger.info(f"  [IVR] Call initiated → {phone}: {call.sid}")
            return True
        except Exception as e:
            logger.error(f"  [IVR] Failed → {phone}: {e}")
            return False

    async def _send_email(self, recipients: List[str],
                           message: str, district: str,
                           hazard: str, probability: float,
                           level: str) -> bool:
        """Send HTML email alert to district officials."""
        if not recipients or not self.smtp_user:
            return False
        try:
            level_color = {'L1':'#10b981','L2':'#eab308',
                           'L3':'#f97316','L4':'#ef4444'}.get(level, '#6b7280')
            html_body = f"""
            <html><body style="font-family:Arial,sans-serif;padding:20px;">
              <div style="border-left:5px solid {level_color};padding-left:15px;">
                <h2 style="color:{level_color};">
                  TRINETRA Alert — {level}: {hazard.upper()}
                </h2>
                <table style="border-collapse:collapse;width:100%;">
                  <tr><td style="padding:6px;font-weight:bold;">District</td>
                      <td style="padding:6px;">{district.title()}</td></tr>
                  <tr><td style="padding:6px;font-weight:bold;">Hazard</td>
                      <td style="padding:6px;">{hazard.upper()}</td></tr>
                  <tr><td style="padding:6px;font-weight:bold;">Risk Probability</td>
                      <td style="padding:6px;color:{level_color};font-weight:bold;">
                        {probability:.1%}</td></tr>
                  <tr><td style="padding:6px;font-weight:bold;">Alert Level</td>
                      <td style="padding:6px;color:{level_color};font-weight:bold;">
                        {level}</td></tr>
                  <tr><td style="padding:6px;font-weight:bold;">Timestamp</td>
                      <td style="padding:6px;">{datetime.utcnow().isoformat()} UTC</td></tr>
                </table>
                <p style="margin-top:15px;">{message}</p>
                <hr/>
                <p style="color:#6b7280;font-size:12px;">
                  Generated by TRINETRA AI AI Powered Geospatial Disaster Intelligence v1.0<br/>
                  SDMA Uttarakhand | NDMA | Geospatial AI Division
                </p>
              </div>
            </body></html>
            """
            msg            = MIMEMultipart('alternative')
            msg['Subject'] = (f'[TRINETRA] {level} — '
                               f'{hazard.upper()} Alert: {district.title()}')
            msg['From']    = self.smtp_user
            msg['To']      = ', '.join(recipients)
            msg.attach(MIMEText(message,   'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.smtp_user, recipients, msg.as_string())

            logger.info(f"  [Email] Sent to {len(recipients)} recipients")
            return True
        except Exception as e:
            logger.error(f"  [Email] Failed: {e}")
            return False

    async def _send_webhooks(self, district: str, hazard: str,
                              level: str, probability: float,
                              contacts: dict) -> bool:
        """
        Push structured alert payload to:
          - SDMA Uttarakhand dashboard
          - NDMA national portal
          - District DM control room system
        """
        if not self.webhook_urls:
            return True  # no-op if no webhooks configured

        payload = {
            'source':           'TRINETRA-v1.0',
            'event_type':       'hazard_alert',
            'district':         district,
            'hazard':           hazard,
            'alert_level':      level,
            'probability':      round(probability, 4),
            'timestamp':        datetime.utcnow().isoformat() + 'Z',
            'population_at_risk': contacts.get('population', 0),
            'relief_camps':     contacts.get('relief_camps', []),
            'geojson_endpoint': f'/v1/riskmap/{district}',
        }

        async with aiohttp.ClientSession() as session:
            for url in self.webhook_urls:
                try:
                    async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers={'Content-Type': 'application/json',
                                 'X-TRINETRA-Version': '1.0.0'}
                    ) as resp:
                        logger.info(f"  [Webhook] {url}: HTTP {resp.status}")
                except asyncio.TimeoutError:
                    logger.warning(f"  [Webhook] Timeout: {url}")
                except Exception as e:
                    logger.error(f"  [Webhook] Error {url}: {e}")
        return True

    async def _log_alert(self, district: str, hazard: str, level: str,
                          probability: float, message: str) -> bool:
        """Append alert to JSONL audit log."""
        log_entry = {
            'ts':          datetime.utcnow().isoformat() + 'Z',
            'district':    district,
            'hazard':      hazard,
            'level':       level,
            'probability': round(probability, 4),
            'message':     message[:200],
        }
        try:
            async with aiofiles.open(self.log_dir / 'alert_log.jsonl', 'a') as f:
                await f.write(json.dumps(log_entry) + '\n')
            return True
        except Exception as e:
            logger.error(f"  [Log] Write failed: {e}")
            return False


# ── Bulk Alert Scheduler ──────────────────────────────────────────────────────
class AlertScheduler:
    """
    Scheduled alert processing for district-wide risk assessments.
    Called by the inference pipeline after each satellite pass.
    """

    def __init__(self, dispatcher: AlertDispatcher):
        self.dispatcher = dispatcher

    async def process_risk_outputs(self, risk_outputs: List[dict]):
        """
        Process list of district risk outputs and dispatch alerts where needed.
        """
        logger.info(f"[Scheduler] Processing {len(risk_outputs)} districts")
        tasks = []
        for output in risk_outputs:
            district = output['district']
            for hz, data in output['hazard_probabilities'].items():
                level = data.get('alert_level', 'NOMINAL')
                if level in ('L2', 'L3', 'L4'):
                    tasks.append(
                        self.dispatcher.dispatch(
                            district    = district,
                            hazard_type = hz,
                            level       = level,
                            probability = data['mean_probability']
                        )
                    )
        if tasks:
            await asyncio.gather(*tasks)
            logger.info(f"[Scheduler] Dispatched {len(tasks)} alerts")
        else:
            logger.info("[Scheduler] No alerts to dispatch (all NOMINAL/L1)")


# ── Alert Dashboard Reporter ──────────────────────────────────────────────────
class AlertDashboardReporter:
    """Generates daily alert summary reports for SDMA."""

    def generate_daily_report(self, log_path: str = './logs/alert_log.jsonl',
                               output_path: str = './reports/daily_report.json'):
        """Aggregate alert statistics for the past 24 hours."""
        import datetime as dt
        from collections import Counter, defaultdict

        cutoff = datetime.utcnow() - dt.timedelta(hours=24)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        entries = []
        if Path(log_path).exists():
            with open(log_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if datetime.fromisoformat(rec['ts'].rstrip('Z')) >= cutoff:
                            entries.append(rec)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

        district_counts  = Counter(e['district'] for e in entries)
        hazard_counts    = Counter(e['hazard']   for e in entries)
        level_counts     = Counter(e['level']    for e in entries)
        max_probs        = defaultdict(float)
        for e in entries:
            key = (e['district'], e['hazard'])
            max_probs[key] = max(max_probs[key], e['probability'])

        report = {
            'report_date':      datetime.utcnow().isoformat() + 'Z',
            'period_hours':     24,
            'total_alerts':     len(entries),
            'districts_affected': dict(district_counts),
            'hazard_breakdown': dict(hazard_counts),
            'level_breakdown':  dict(level_counts),
            'peak_risks':       {f"{k[0]}_{k[1]}": round(v, 4)
                                  for k, v in max_probs.items()},
            'critical_events':  [e for e in entries if e['level'] == 'L4'],
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"[Report] Daily report → {output_path} "
                    f"({len(entries)} alerts)")
        return report


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import asyncio

    async def demo():
        print("TRINETRA — Alert System Demo")
        print("=" * 50)
        dispatcher = AlertDispatcher()
        print("\nSimulating L4 Landslide alert for Chamoli...")
        await dispatcher.dispatch(
            district    = 'chamoli',
            hazard_type = 'landslide',
            level       = 'L4',
            probability = 0.91
        )
        print("\nSimulating L3 Flood alert for Rudraprayag...")
        await dispatcher.dispatch(
            district    = 'rudraprayag',
            hazard_type = 'flood',
            level       = 'L3',
            probability = 0.79
        )
        reporter = AlertDashboardReporter()
        report = reporter.generate_daily_report()
        print(f"\nDaily report: {report['total_alerts']} alerts in last 24h")
        print("\n✓ Alert system demo complete.")

    asyncio.run(demo())
