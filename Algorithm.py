import socket
import sys
import time
import http.client
import urllib.parse

def stream_to_http_server(host='localhost', port=8080, interval=1):
    """
    Streams data to an HTTP server using GET requests.
    
    Args:
        host (str): The server hostname or IP address
        port (int): The port number to connect to
        interval (float): Time in seconds between sending data
    """
    # Generate and display URL for the port
    url = f"http://{host}:{port}"
    print(f"Streaming URL: {url}")
    print(f"Starting streaming to {host}:{port}")
    
    count = 0
    while True:
        try:
            # Get current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Create a message with timestamp and sequence number
            message = f"DATA PACKET #{count} - {timestamp}"
            
            # URL encode the message for GET parameter
            encoded_message = urllib.parse.quote(message)
            
            # Send HTTP GET request
            conn = http.client.HTTPConnection(host, port)
            conn.request("GET", f"/?data={encoded_message}")
            
            # Get the response
            response = conn.getresponse()
            print(f"Sent: {message} - Response: {response.status} {response.reason}")
            
            # Close the connection
            conn.close()
            
            # Wait for the specified interval
            time.sleep(interval)
            count += 1
            
        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            # Wait before retrying
            time.sleep(interval)

if __name__ == "__main__":
    # Default parameters
    host = 'localhost'
    port = 8080
    interval = 1
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    if len(sys.argv) > 3:
        interval = float(sys.argv[3])
    
    # Start streaming
    stream_to_http_server(host, port, interval)