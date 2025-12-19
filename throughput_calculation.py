import json
from datetime import datetime

def calculate_throughput(results_data):
    """
    Calculate tokens per second throughput from streaming response data
    """
    if not results_data or 'results' not in results_data:
        return 0
    
    results = results_data['results']
    
    # Group by request_idx to handle multiple requests
    requests = {}
    for result in results:
        request_idx = result.get('request_idx', 0)
        if request_idx not in requests:
            requests[request_idx] = []
        requests[request_idx].append(result)
    
    total_tokens = 0
    total_time_ms = 0
    
    for request_idx, request_results in requests.items():
        if not request_results:
            continue
            
        # Sort by response_idx to ensure chronological order
        request_results.sort(key=lambda x: x.get('response_idx', 0))
        
        # Get start and end timestamps for this request
        start_time = request_results[0]['timestamp']
        end_time = request_results[-1]['timestamp']
        
        # Calculate duration in seconds
        duration_ms = (end_time - start_time) / 1_000_000  # Convert from nanoseconds to milliseconds
        duration_seconds = duration_ms / 1000  # Convert to seconds
        
        # Count tokens in this request
        tokens_in_request = len(request_results)
        
        total_tokens += tokens_in_request
        total_time_ms += duration_ms
    
    # Calculate overall throughput
    if total_time_ms > 0:
        total_time_seconds = total_time_ms / 1000
        tokens_per_second = total_tokens / total_time_seconds
    else:
        tokens_per_second = 0
    
    return {
        'total_requests': len(requests),
        'total_tokens': total_tokens,
        'total_time_seconds': total_time_ms / 1000,
        'tokens_per_second': tokens_per_second,
        'avg_tokens_per_request': total_tokens / len(requests) if requests else 0,
        'avg_time_per_request_ms': total_time_ms / len(requests) if requests else 0
    }

# Load and process the data
with open('results.json', 'r') as f:
    data = json.load(f)

throughput = calculate_throughput(data)

print("Throughput Analysis:")
print(f"Total requests: {throughput['total_requests']}")
print(f"Total tokens: {throughput['total_tokens']}")
print(f"Total time: {throughput['total_time_seconds']:.3f} seconds")
print(f"Tokens per second: {throughput['tokens_per_second']:.2f}")
print(f"Average tokens per request: {throughput['avg_tokens_per_request']:.2f}")
print(f"Average time per request: {throughput['avg_time_per_request_ms']:.2f} ms")

# Additional detailed analysis per request
print("\nDetailed per-request analysis:")
requests = {}
for result in data['results']:
    request_idx = result.get('request_idx', 0)
    if request_idx not in requests:
        requests[request_idx] = []
    requests[request_idx].append(result)

for request_idx in sorted(requests.keys()):
    request_results = requests[request_idx]
    request_results.sort(key=lambda x: x.get('response_idx', 0))
    
    start_time = request_results[0]['timestamp']
    end_time = request_results[-1]['timestamp']
    duration_ms = (end_time - start_time) / 1_000_000
    
    print(f"Request {request_idx}: {len(request_results)} tokens in {duration_ms:.2f} ms "
          f"({len(request_results)/(duration_ms/1000):.2f} tokens/sec)")