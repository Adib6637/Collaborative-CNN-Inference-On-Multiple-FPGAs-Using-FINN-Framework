import socket
import time

# Server setup
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 9009      # Arbitrary port number

# Number of bytes to receive (adjustable)
data_size = 10 * 1024 * 1024  # Example: 10 MB

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    with conn:
        print(f"Connection established with {addr}")
        start_time = time.time()
        
        received_data = 0
        while received_data < data_size:
            data = conn.recv(4096)
            if not data:
                break
            received_data += len(data)

        end_time = time.time()
        print(f"Received {received_data} bytes")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Transfer speed: {received_data / (end_time - start_time) / (1024 * 1024):.2f} MB/s")