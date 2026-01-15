#!/usr/bin/env python3
"""Convert OpenHands output history to a readable HTML format with Microagent-like UI."""

import argparse
import json
import os
import html

# --- CSS / HTML Template ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenHands Trace: {instance_id}</title>
    <style>
        :root {{
            --bg-color: #0d0d0d;
            --card-bg: #171717;
            --text-primary: #e5e5e5;
            --text-secondary: #a3a3a3;
            --border-color: #262626;
            --accent-green: #4ade80;
            --code-bg: #262626;
            --timeline-color: #404040;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
        }}

        .container {{
            width: 100%;
            max-width: 900px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .header {{
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .metadata-section details {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .metadata-section summary {{
            padding: 10px 15px;
            cursor: pointer;
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .timeline {{
            position: relative;
            padding-left: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .timeline::before {{
            content: '';
            position: absolute;
            top: 0;
            bottom: 0;
            left: 6px;
            width: 2px;
            background-color: var(--timeline-color);
            z-index: 0;
        }}

        .step-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            z-index: 1;
        }}

        /* User Message Style */
        .user-message {{
            align-self: flex-end;
            background-color: #2b2b2b;
            border: 1px solid #404040;
            border-radius: 12px;
            padding: 12px 16px;
            max-width: 80%;
            position: relative;
            margin-bottom: 10px;
        }}
        
        .user-message-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Step Details & Summary */
        details.step-details {{
            width: 100%;
        }}

        summary.step-summary {{
            list-style: none; /* Hide default triangle */
            padding: 12px 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
        }}
        
        summary.step-summary::-webkit-details-marker {{
            display: none;
        }}

        .summary-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
            font-size: 0.95rem;
        }}

        .chevron {{
            width: 16px;
            height: 16px;
            fill: none;
            stroke: var(--text-secondary);
            stroke-width: 2;
            transition: transform 0.2s;
        }}

        details[open] .chevron {{
            transform: rotate(180deg);
        }}

        .status-icon {{
            color: var(--accent-green);
        }}

        .step-content {{
            padding: 16px;
            background-color: #0f0f0f;
            border-top: 1px solid var(--border-color);
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            white-space: pre-wrap;
            line-height: 1.5;
            color: #d4d4d4;
        }}
        
        /* Specific content highlighting */
        .diff-add {{ color: #4ade80; }}
        .diff-del {{ color: #f87171; }}
        .diff-header {{ color: #60a5fa; font-weight: bold; }}
        
        .cmd-input {{ color: #fbbf24; }}
        
        .thought-text {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-secondary);
            margin-bottom: 10px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{instance_id}</h1>
        </div>
        
        <!-- Metadata section removed by user request -->

        <div class="timeline">
            {trace_html}
        </div>
    </div>
</body>
</html>
"""

ICON_CHECK = """<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="status-icon"><polyline points="20 6 9 17 4 12"></polyline></svg>"""
ICON_CHEVRON = """<svg viewBox="0 0 24 24" class="chevron" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>"""
ICON_USER = """<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>"""

def escape_content(text):
    return html.escape(str(text))

def format_diff(diff_text):
    lines = diff_text.split('\n')
    formatted = []
    for line in lines:
        if line.startswith('+'):
            formatted.append(f'<span class="diff-add">{escape_content(line)}</span>')
        elif line.startswith('-'):
            formatted.append(f'<span class="diff-del">{escape_content(line)}</span>')
        elif line.startswith('@@') or line.startswith('diff'):
            formatted.append(f'<span class="diff-header">{escape_content(line)}</span>')
        else:
            formatted.append(escape_content(line))
    return '\n'.join(formatted)


def get_observation_content(observation_item):
    """Extract text content from an Observation item (Event or inner dict)."""
    if not observation_item:
        return ""
    
    # Unwrap 'ObservationEvent' if valid
    real_obs = observation_item
    if 'observation' in observation_item and isinstance(observation_item['observation'], dict):
        real_obs = observation_item['observation']
        
    content = real_obs.get('content', '')
    
    # Handle list of text blocks (e.g. OpenHands)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                text_parts.append(block['text'])
            else:
                text_parts.append(str(block))
        return "\n".join(text_parts)
        
    return str(content)

def render_step(title, content, is_open=False, is_user=False):
    if is_user:
        return f"""<div class="user-message"><div class="user-message-label">User</div><div style="white-space: pre-wrap;">{content}</div></div>"""
    
    open_attr = "open" if is_open else ""
    return f"""<div class="step-card"><details class="step-details" {open_attr}><summary class="step-summary"><div class="summary-left"><span>{title}</span></div><div style="display:flex; gap:10px; align-items:center;">{ICON_CHECK}{ICON_CHEVRON}</div></summary><div class="step-content">{content}</div></details></div>"""

def convert_history_to_html(history):
    html_out = ""
    # Process sequential pairs
    i = 0
    while i < len(history):
        item = history[i]
        
        # Skip ConversationStateUpdateEvent or SystemPromptEvent if desired
        # The user specifically requested hiding ConversationStateUpdateEvent
        if isinstance(item, dict) and item.get('kind') == 'ConversationStateUpdateEvent':
            i += 1
            continue

        # Check for generic ActionEvent wrapper
        # If item is ActionEvent, sometimes it wraps the real action details in 'action' dict (handled in process_event_pair)
        
        # Try to look ahead for observation if current is action
        action = item
        observation = None
        
        # Simple heuristic: if next event is Observation and relates to this action?
        if i + 1 < len(history) and not isinstance(history[i+1], list):
            next_item = history[i+1]
            if 'observation_type' in next_item or next_item.get('kind') == 'ObservationEvent':
                observation = next_item
                # Skip next item as we consumed it
                i += 1
        
        # Check if this is effectively the last item
        is_last = is_tail_effectively_empty(history, i + 1)
        
        # If current item is actually an Observation (orphan), handle it?
        if 'observation_type' in action or action.get('kind') == 'ObservationEvent':
            # Orphan observation
            obs_content = get_observation_content(action)
            html_out += render_step(f"Observation: {action.get('observation_type', 'Output')}", escape_content(obs_content), is_open=is_last)
        else:
            html_out += process_event_pair(action, observation, is_last=is_last)
        i += 1

    return html_out



def get_thought_content(props):
    """Try to extract meaningful thought/reasoning from action properties."""
    # 1. Try 'reasoning_content' which usually has the raw text
    rc = props.get('reasoning_content')
    if rc and isinstance(rc, str) and rc.strip():
        return rc
        
    # 2. Try 'thought'
    th = props.get('thought')
    if isinstance(th, str):
        return th
        
    if isinstance(th, list):
        # Join text fields from list of dicts
        parts = []
        for block in th:
            if isinstance(block, dict) and 'text' in block:
                parts.append(str(block['text']))
            elif isinstance(block, str):
                parts.append(block)
        joined = "\n".join(parts).strip()
        if joined:
            return joined
            
    return ""

def is_tail_effectively_empty(history, start_idx):
    """Check if the remaining history contains any renderable events."""
    for j in range(start_idx, len(history)):
        it = history[j]
        # We skip ConversationStateUpdateEvent
        if isinstance(it, dict) and it.get('kind') == 'ConversationStateUpdateEvent':
            continue
        return False
    return True

def process_event_pair(action, observation=None, is_last=False):
    # Support both flat and nested action structures
    if 'action' in action and isinstance(action['action'], dict):
        # Nested structure (e.g. from OpenHands output.jsonl)
        real_action = action['action']
        act_type = real_action.get('kind', '') or real_action.get('action', 'Unknown')
        # Map common kinds
        if not act_type.endswith('Action'):
             act_type += 'Action' 
        
        # Merge properties for easy access
        # but keep original available
        action_props = real_action
        
        # Merge top-level properties (like thought/reasoning_content) into action_props if missing
        # This is critical because 'reasoning_content' is often at the top level event, not inside 'action'
        for k, v in action.items():
            if k not in action_props and k in ['thought', 'reasoning_content']:
                action_props[k] = v
                
    else:
        # Flat structure
        act_type = action.get('action_type', '')
        if not act_type:
            act_type = action.get('type', 'Unknown')
        action_props = action
    
    # 1. User Message
    if act_type == 'MessageAction' and action.get('source') == 'user':
         return render_step("", escape_content(action.get('content', '')), is_user=True)

    title = act_type
    content = ""
    # Default is folded, unless it is the last item
    is_open = is_last

    # Title Logic
    kind = action_props.get('kind', '')
    
    # Common Logic: Reasoning (Thought) attached to generic actions
    thought_html = ""
    thought_text = get_thought_content(action_props)
    
    if thought_text:
        thought_html = f"""<details class="step-details" style="margin-bottom: 8px;"><summary style="cursor:pointer; color: var(--text-secondary); font-size: 0.85rem;">Reasoning</summary><div class='thought-text' style="margin-top: 5px; padding-left: 10px; border-left: 2px solid var(--border-color);">{escape_content(thought_text)}</div></details>"""

    if act_type == 'CmdRunAction' or kind == 'run' or kind == 'TerminalAction':
        cmd = action_props.get('command', '')
        
        # User request: Use summary as title, display command inside block
        # Valid summary is usually at event level (action) or props level
        summary_text = action.get('summary') or action_props.get('summary')
        if summary_text:
            title = escape_content(summary_text)
            # Display command in content
            content += f"<div class='cmd-text' style='font-family:monospace; background:#222; padding:8px; border-radius:6px; margin-bottom:8px;'>{html.escape(cmd)}</div>"
        else:
            title = f"Ran {html.escape(cmd)}"
        
        content += thought_html
        
        # Output label + content
        if observation:
            obs_content = get_observation_content(observation)
            content += f"<div>Output:</div><div style='margin-top:4px'>{escape_content(obs_content)}</div>"

    elif act_type == 'FileWriteAction' or kind == 'file_write' or (kind == 'FileEditorAction' and action_props.get('command') in ['write', 'create', 'str_replace']):
        path = action_props.get('path', '')
        cmd = action_props.get('command', 'edit')
        title = f"Edited {html.escape(path)}"
        if cmd == 'create':
             title = f"Created {html.escape(path)}"
        
        content += thought_html
        
        file_content = action_props.get('file_text', '') or action_props.get('content', '')
        content += format_diff(file_content)
        
        if observation:
             obs_content = get_observation_content(observation)
             if obs_content:
                 content += f"<div style='margin-top:10px; border-top:1px solid #333; padding-top:5px'>Output: {escape_content(obs_content)}</div>"
    
    elif act_type == 'FileReadAction' or kind == 'file_read' or (kind == 'FileEditorAction' and action_props.get('command') == 'view'):
        path = action_props.get('path', '')
        title = f"Viewed {html.escape(path)}"
        content += thought_html
        if observation:
            obs_content = get_observation_content(observation)
            content += f"<div>Output:</div><div style='margin-top:4px'>{escape_content(obs_content)}</div>"

    elif act_type == 'ThinkAction' or kind == 'think' or kind == 'ThinkAction': 
        title = "Microagent ready"
        # User request: Get .action.thought as main message
        # We prefer the raw 'thought' field from the action dict
        thought_body = action_props.get('thought')
        
        # Fallback to helper if direct access is missing/complex
        if not thought_body:
             thought_body = get_thought_content(action_props)
        elif isinstance(thought_body, list):
             # Just in case it's a list, flatten it similarly
             thought_body = get_thought_content({'thought': thought_body})
             
        content = f"<div class='thought-text' style='white-space: pre-wrap;'>{escape_content(str(thought_body))}</div>"
    
    elif act_type == 'IPythonRunCellAction' or kind == 'run_cell':
        code = action_props.get('code', '')
        title = "Ran Python Code"
        content += thought_html
        content += f"{escape_content(code)}\n"
        if observation:
             obs_content = get_observation_content(observation)
             content += "\n<div>Output:</div><div style='margin-top:4px'>" + escape_content(obs_content) + "</div>"
             
    elif kind == 'FinishAction':
        title = "Task Finished"
        content += thought_html
        # User request: get .action.message
        msg = action_props.get('message', 'Agent submitted the task.')
        content += f"<div style='margin-top:5px'>{escape_content(msg)}</div>"
        
    elif kind == 'SystemPromptEvent':
        title = "System Prompt"
        # Extract system prompt text
        sys_text = action_props.get('system_prompt', "System defined tools and prompt.")
        if isinstance(sys_text, dict) and 'text' in sys_text:
             sys_text = sys_text['text']
        content = f"<div style='white-space: pre-wrap;'>{escape_content(str(sys_text))}</div>"
        
    elif kind == 'MessageEvent':
        title = "User Message"
        if 'llm_message' in action_props and 'content' in action_props['llm_message']:
            msg_content = action_props['llm_message']['content']
            if isinstance(msg_content, list):
                 text_parts = []
                 for block in msg_content:
                     if isinstance(block, dict) and 'text' in block:
                         text_parts.append(block['text'])
                     else:
                         text_parts.append(str(block))
                 content = "\n".join(text_parts)
            else:
                 content = str(msg_content)
            
            content = f"<div style='white-space: pre-wrap;'>{escape_content(content)}</div>"
            is_open = True

    else:
        # Generic fallback - NO JSON DUMP
        title = f"Action: {kind or act_type}"
        content += thought_html
        # Try to find some content
        content_txt = action_props.get('content') or ""
        if content_txt:
            content += escape_content(str(content_txt))
        else:
            if not thought_html:
                content += "No details available."
            
        if observation:
             obs_content = get_observation_content(observation)
             if obs_content:
                 content += f"\n<div>Output:</div><div style='margin-top:4px'>{escape_content(obs_content)}</div>"

    return render_step(title, content, is_open=is_open)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('oh_output_file', type=str, help='Path to the OpenHands output.jsonl file')
    args = parser.parse_args()

    output_viz_folder = args.oh_output_file.replace('.jsonl', '.html.viz')
    os.makedirs(output_viz_folder, exist_ok=True)
    print(f'Converting {args.oh_output_file} to HTML files in {output_viz_folder}')

    try:
        with open(args.oh_output_file, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_idx + 1}: {e}")
                    continue

                instance_id = row.get('instance_id', 'unknown_instance')
                filename = f'{instance_id}.html'
                filepath = os.path.join(output_viz_folder, filename)
                
                metadata_str = json.dumps(row.get('metadata', {}), indent=2)
                trace_html = ""
                if 'history' in row and row['history']:
                    trace_html = convert_history_to_html(row['history'])
                
                
                full_html = HTML_TEMPLATE.format(
                    instance_id=instance_id,
                    # metadata_json=escape_content(metadata_str),
                    trace_html=trace_html
                )
                
                with open(filepath, 'w') as f_out:
                    f_out.write(full_html)
                    
    except Exception as e:
        print(f"Error reading {args.oh_output_file}: {e}")

if __name__ == '__main__':
    main()
