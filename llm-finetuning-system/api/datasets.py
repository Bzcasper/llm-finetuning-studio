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
            # Try to connect to Modal and list datasets
            modal_token_id = os.environ.get("MODAL_TOKEN_ID")
            modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
            
            if modal_token_id and modal_token_secret:
                try:
                    import modal
                    
                    # Set up Modal credentials
                    os.environ["MODAL_TOKEN_ID"] = modal_token_id
                    os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret
                    
                    # Try to lookup the app and call the function
                    app = modal.App.lookup("llm-finetuner", create_if_missing=False)
                    result = app.list_models_and_datasets.remote()
                    
                    response = {
                        "datasets": result.get("datasets", []),
                        "source": "modal",
                        "timestamp": time.time()
                    }
                    
                except Exception as e:
                    # Fallback to simulated data
                    response = {
                        "datasets": [
                            {
                                "name": "dummy_dataset.jsonl",
                                "path": "/vol/dummy_dataset.jsonl",
                                "size": "1.2 KB",
                                "type": "jsonl",
                                "created": time.time() - 86400
                            },
                            {
                                "name": "alpaca_sample.json",
                                "path": "/vol/alpaca_sample.json", 
                                "size": "15.3 MB",
                                "type": "json",
                                "created": time.time() - 172800
                            }
                        ],
                        "source": "simulated",
                        "error": str(e),
                        "timestamp": time.time()
                    }
            else:
                # No credentials, return empty list
                response = {
                    "datasets": [],
                    "source": "no_credentials",
                    "error": "Modal credentials not configured",
                    "timestamp": time.time()
                }
                
        except ImportError:
            # Modal not available in serverless environment
            response = {
                "datasets": [
                    {
                        "name": "sample_dataset.jsonl",
                        "path": "local/sample_dataset.jsonl",
                        "size": "2.1 KB", 
                        "type": "jsonl",
                        "created": time.time() - 3600
                    }
                ],
                "source": "local_simulation",
                "error": "Modal not available in serverless environment",
                "timestamp": time.time()
            }
        except Exception as e:
            response = {
                "datasets": [],
                "source": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

