#!/usr/bin/env python3
"""
GP MLOps Dashboard Frontend Server
Serves the cool dashboard on port 2222
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import threading
import webbrowser
import time

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def start_frontend_server():
    """Start the frontend server on port 2222"""
    
    # Change to dashboard directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    server_address = ('', 2222)
    httpd = HTTPServer(server_address, CORSHTTPRequestHandler)
    
    print("🎨 GP MLOps Dashboard Frontend")
    print("=" * 50)
    print(f"🔗 Dashboard URL: http://localhost:2222")
    print(f"📊 Backend API: http://localhost:2233")
    print("=" * 50)
    print("✅ Frontend server started on port 2222")
    print("💡 Make sure backend is running on port 2233")
    print()
    
    # Auto-open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:2222')
            print("🌐 Opened dashboard in your default browser")
        except:
            print("🌐 Please open http://localhost:2222 in your browser")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    start_frontend_server()