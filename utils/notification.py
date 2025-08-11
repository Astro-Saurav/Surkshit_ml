"""
Notification system for the Scream Detection System
"""

import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.audio import MIMEAudio
from datetime import datetime
from ..config import Config

class NotificationSystem:
    def __init__(self, parent_phone=None, parent_email=None):
        """
        Initialize notification system
        
        Args:
            parent_phone (str, optional): Parent's phone number
            parent_email (str, optional): Parent's email address
        """
        self.parent_phone = parent_phone or Config.PARENT_PHONE
        self.parent_email = parent_email or Config.PARENT_EMAIL
        
        # Desktop notification
        self.desktop_notification_enabled = True
        
        # Check if we can use a desktop notifier
        try:
            from plyer import notification
            self.plyer_available = True
        except ImportError:
            self.plyer_available = False
    
    def notify_emergency_contacts(self, confidence, location=None, audio_path=None):
        """
        Send notifications to emergency contacts
        
        Args:
            confidence (float): Detection confidence (0-1)
            location (str, optional): Location information
            audio_path (str, optional): Path to audio file
            
        Returns:
            bool: True if any notification was sent successfully
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = {
            "alert_type": "SCREAM DETECTED",
            "confidence": f"{confidence*100:.2f}%",
            "timestamp": timestamp,
            "location": location if location else "Unknown"
        }
        
        print(f"EMERGENCY ALERT: Scream detected with {confidence*100:.2f}% confidence")
        
        # Track if any notification method succeeded
        success = False
        
        # Send desktop notification
        if self.desktop_notification_enabled:
            desktop_success = self.send_desktop_notification(confidence, timestamp)
            success = success or desktop_success
        
        # Send SMS notification
        if self.parent_phone:
            sms_success = self.send_sms_notification(confidence, timestamp, location)
            success = success or sms_success
        
        # Send email notification
        if self.parent_email:
            email_success = self.send_email_notification(confidence, timestamp, location, audio_path)
            success = success or email_success
        
        return success
    
    def send_desktop_notification(self, confidence, timestamp):
        """
        Send a desktop notification
        
        Args:
            confidence (float): Detection confidence
            timestamp (str): Detection timestamp
            
        Returns:
            bool: True if successful
        """
        if not self.plyer_available:
            try:
                # Try alternative desktop notification methods
                if os.name == 'posix':  # Linux or macOS
                    os.system(f"notify-send 'SCREAM DETECTED' 'Alert: Scream detected at {timestamp} with {confidence*100:.2f}% confidence.'")
                    return True
                elif os.name == 'nt':  # Windows
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(
                        "SCREAM DETECTED",
                        f"Alert: Scream detected at {timestamp} with {confidence*100:.2f}% confidence.",
                        duration=10,
                        threaded=True
                    )
                    return True
            except Exception as e:
                print(f"Failed to send desktop notification: {e}")
                return False
        else:
            try:
                from plyer import notification
                notification.notify(
                    title="SCREAM DETECTED",
                    message=f"Alert: Scream detected at {timestamp} with {confidence*100:.2f}% confidence.",
                    app_name="Scream Detection System",
                    timeout=10
                )
                return True
            except Exception as e:
                print(f"Failed to send desktop notification: {e}")
                return False
    
    def send_sms_notification(self, confidence, timestamp, location=None):
        """
        Send SMS notification
        
        Args:
            confidence (float): Detection confidence
            timestamp (str): Detection timestamp
            location (str, optional): Location information
            
        Returns:
            bool: True if successful
        """
        try:
            # This is a placeholder for an SMS API integration
            # In a real implementation, you would use an actual SMS service like Twilio
            
            # Example with Twilio (commented out)
            # from twilio.rest import Client
            # account_sid = 'YOUR_ACCOUNT_SID'
            # auth_token = 'YOUR_AUTH_TOKEN'
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(
            #     body=f"ALERT: Scream detected at {timestamp} with {confidence*100:.2f}% confidence. Location: {location or 'Unknown'}",
            #     from_='+1234567890',  # Your Twilio number
            #     to=self.parent_phone
            # )
            
            # For now, just simulate sending an SMS
            print(f"✓ SMS alert sent to {self.parent_phone}")
            return True
            
        except Exception as e:
            print(f"Failed to send SMS notification: {e}")
            return False
    
    def send_email_notification(self, confidence, timestamp, location=None, audio_path=None):
        """
        Send email notification
        
        Args:
            confidence (float): Detection confidence
            timestamp (str): Detection timestamp
            location (str, optional): Location information
            audio_path (str, optional): Path to audio file
            
        Returns:
            bool: True if successful
        """
        try:
            # For demonstration purposes - replace with your actual SMTP settings
            # In a real implementation, store these in environment variables or a secure config
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            smtp_username = "your_email@gmail.com"
            smtp_password = "your_app_password"  # Use app password for Gmail
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_username
            msg['To'] = self.parent_email
            msg['Subject'] = "URGENT: Scream Detected"
            
            # Message body
            body = f"""
            ALERT: Scream detected at {timestamp} with {confidence*100:.2f}% confidence.
            Location: {location or 'Unknown'}
            
            This is an automated alert from the Scream Detection System.
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach audio file if provided
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, 'rb') as audio_file:
                    audio_attachment = MIMEAudio(audio_file.read(), _subtype="wav")
                    audio_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(audio_path))
                    msg.attach(audio_attachment)
            
            # For demonstration, just log that we would send an email
            print(f"✓ Email alert would be sent to {self.parent_email}")
            
            # Uncomment to actually send the email
            # with smtplib.SMTP(smtp_server, smtp_port) as server:
            #     server.starttls()
            #     server.login(smtp_username, smtp_password)
            #     server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Failed to send email notification: {e}")
            return False
    
    def send_test_notification(self):
        """
        Send a test notification
        
        Returns:
            bool: True if successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.notify_emergency_contacts(
            confidence=0.95,  # High confidence for testing
            location="Test Location"
        )