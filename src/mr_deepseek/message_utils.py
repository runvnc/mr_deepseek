def strip_cache_control(content):
    """Remove cache_control for comparison purposes"""
    if isinstance(content, dict):
        return {k: v for k, v in content.items() if k != 'cache_control'}
    return content

def compare_messages(previous_messages, current_messages):
    """
    Compare two sets of messages to find which ones have changed.
    Returns indices of changed messages in current_messages.
    Ignores cache_control differences.
    
    Args:
        previous_messages: List of message dicts from previous call
        current_messages: List of message dicts from current call
        
    Returns:
        changed_indices: List of indices where messages differ
    """
    changed_indices = []
    
    # Handle empty previous messages
    if not previous_messages:
        print('\033[94m[CACHE] First run - all messages are new\033[0m')
        return list(range(len(current_messages)))
        
    # Compare messages
    for i, curr_msg in enumerate(current_messages):
        # If message is beyond previous length, it's new
        if i >= len(previous_messages):
            print(f'\033[92m[CACHE] New message at index {i}:\n+ {curr_msg.get("content", "")}\033[0m')
            changed_indices.append(i)
            continue
            
        prev_msg = previous_messages[i]
        
        # Compare role
        if curr_msg.get('role') != prev_msg.get('role'):
            print(f'\033[93m[CACHE] Role changed at index {i}:\n- {prev_msg.get("role")}\n+ {curr_msg.get("role")}\033[0m')
            changed_indices.append(i)
            continue
            
        # Compare content
        curr_content = curr_msg.get('content', '')
        prev_content = prev_msg.get('content', '')
        
        # Handle string content
        if isinstance(curr_content, str) and isinstance(prev_content, str):
            if curr_content != prev_content:
                print(f'\033[91m[CACHE] Content changed at index {i}:\n- {prev_content}\n+ {curr_content}\033[0m')
                changed_indices.append(i)
                continue
                
        # Handle list content
        elif isinstance(curr_content, list) and isinstance(prev_content, list):
            if len(curr_content) != len(prev_content):
                print(f'\033[91m[CACHE] Content list length changed at index {i}:\n- {len(prev_content)} items\n+ {len(curr_content)} items\033[0m')
                changed_indices.append(i)
                continue
                
            # Compare each content item without cache_control
            for j, (curr_item, prev_item) in enumerate(zip(curr_content, prev_content)):
                curr_stripped = strip_cache_control(curr_item)
                prev_stripped = strip_cache_control(prev_item)
                if curr_stripped != prev_stripped:
                    print(f'\033[91m[CACHE] Content list item {j} changed at index {i}:\n- {prev_stripped}\n+ {curr_stripped}\033[0m')
                    changed_indices.append(i)
                    break
                    
        # Different content types
        else:
            print(f'\033[93m[CACHE] Content type changed at index {i}:\n- {type(prev_content)}\n+ {type(curr_content)}\033[0m')
            changed_indices.append(i)
    
    if not changed_indices:
        print('\033[92m[CACHE] No changes detected in messages\033[0m')
    else:
        print(f'\033[94m[CACHE] Changed message indices: {changed_indices}\033[0m')
            
    return changed_indices