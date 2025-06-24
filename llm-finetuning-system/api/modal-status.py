from http.server import BaseHTTPRequestHandler
import json
import os
import time

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            # Check Modal connection
            modal_token_id = os.environ.get("MODAL_TOKEN_ID")
            modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
            modal_profile = os.environ.get("MODAL_PROFILE", "ai-tool-pool")
            
            connected = False
            error_message = None
            app_status = "unknown"
            
            if modal_token_id and modal_token_secret:
                try:
                    import modal
                    
                    # Set environment variables for Modal
                    os.environ["MODAL_TOKEN_ID"] = modal_token_id
                    os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret
                    
                    # Try to lookup the app
                    app = modal.App.lookup("llm-finetuner", create_if_missing=False)
                    connected = True
                    app_status = "deployed"
                    
                except Exception as e:
                    error_message = str(e)
                    if "No such environment" in str(e):
                        app_status = "environment_not_found"
                    elif "not found" in str(e):
                        app_status = "app_not_deployed"
                    else:
                        app_status = "connection_error"
            else:
                error_message = "Modal credentials not configured"
                app_status = "credentials_missing"
            
            response = {
                "connected": connected,
                "environment": modal_profile,
                "app_status": app_status,
                "error": error_message,
                "timestamp": time.time(),
                "credentials_configured": bool(modal_token_id and modal_token_secret)
            }
            
        except ImportError:
            response = {
                "connected": False,
                "environment": "unknown",
                "app_status": "modal_not_installed",
                "error": "Modal library not available in serverless environment",
                "timestamp": time.time(),
                "credentials_configured": False
            }
        except Exception as e:
            response = {
                "connected": False,
                "environment": "unknown", 
                "app_status": "error",
                "error": str(e),
                "timestamp": time.time(),
                "credentials_configured": False
            }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

