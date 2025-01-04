import socket
import time

# Server IP and port
SERVER_IP = '192.168.178.38'  # Replace with the server's IP
PORT = 9009

# Number of bytes to send (adjustable)
data_size = 10 * 1024 * 1024  # Example: 10 MB
data_to_send = b'0' * data_size

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((SERVER_IP, PORT))
    print(f"Connected to server at {SERVER_IP}:{PORT}")

    start_time = time.time()
    client_socket.sendall(data_to_send)
    end_time = time.time()

    print(f"Sent {data_size} bytes")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Transfer speed: {data_size / (end_time - start_time) / (1024 * 1024):.2f} MB/s")
