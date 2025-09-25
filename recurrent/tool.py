# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional, Type, List, Union, Dict, Callable
import jsonschema
import re
import json
from transformers.tokenization_utils_base import get_json_schema

from collections import namedtuple
ToolCall = namedtuple('ToolCall', ['name', 'args'])

class ToolSchema():
    INVALID_TOOL = "__INVALID_TOOL__"
    @classmethod
    def invalid(cls, msg):
        return ToolCall(cls.INVALID_TOOL, {"msg": msg})

    @staticmethod
    def is_tool_schema(obj: dict) -> bool:
        # Adapted from agent-r1 https://github.com/0russwest0/Agent-R1/blob/main/agent_r1/tool/base.py
        """
        Check if obj is a valid JSON schema describing a tool compatible with OpenAI's tool calling.
        Example valid schema:
        {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
            },
            "required": ["location"]
        }
        }
        """
        try:
            assert set(obj.keys()) == {'name', 'description', 'parameters'}
            assert isinstance(obj['name'], str)
            assert obj['name'].strip()
            assert isinstance(obj['description'], str)
            assert isinstance(obj['parameters'], dict)

            assert set(obj['parameters'].keys()) == {'type', 'properties', 'required'}
            assert obj['parameters']['type'] == 'object'
            assert isinstance(obj['parameters']['properties'], dict)
            assert isinstance(obj['parameters']['required'], list)
            assert set(obj['parameters']['required']).issubset(set(obj['parameters']['properties'].keys()))
        except AssertionError:
            return False
        try:
            jsonschema.validate(instance={}, schema=obj['parameters'])
        except jsonschema.exceptions.SchemaError:
            return False
        except jsonschema.exceptions.ValidationError:
            pass
        return True

    def __init__(self, 
        name: str = '',
        description: str = '',
        parameters: Dict[str, Any] = None,
    ):
        if not name:
            raise ValueError(f'missing tool name: {self}')
        self.name = name
        self.description = description
        if not parameters:
            self.parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
        else:
            self.parameters = parameters
        if not self.is_tool_schema(self.schema['function']):
            raise ValueError(f'not a valid openai-compatible schema: {self}')
    
    @classmethod
    def from_function(cls, func: Callable):
        schema = get_json_schema(func)
        return cls(**schema)

    @property
    def schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
        }
        }

    def __str__(self):
        def param_string(k, v) -> str:
            return f"- {k} ({v['type']}): {v['description']}"
        param_strings = "\n".join([param_string(k, v) for k, v in self.parameters['properties'].items()])
        
        return f"""
## Tool {self.name}
Description: {self.description}

Parameters (required: {self.parameters['required']}):
{param_strings}
""".lstrip()


def toolcall_system_prompt(tools: List[ToolSchema]):
    # Qwen2TokenizerFast.chat_template
    return """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>""".format(
        tools='\n'.join([
            json.dumps(t.schema, ensure_ascii=False)
            for t in tools
        ])
    )

def toolcall_extract(text: str) -> Dict:
    # adapted from a previous version of agent-r1  https://github.com/0russwest0/Agent-R1

    # Regular expression to extract tool call <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
    if not tool_call_match:
        return None
    try:
        tool_call_json = tool_call_match.group(1).strip()
        tool_call_data = json.loads(tool_call_json)
        # Extract tool name and arguments
        if "name" not in tool_call_data:
            return ToolSchema.invalid(f"tool name not found")
        return ToolCall(tool_call_data["name"], tool_call_data.get("arguments", {}))
    except json.JSONDecodeError:
        return ToolSchema.invalid(f"invalid json")

def merge_system_prompt(msg: List[Dict[str, str]], system_prompt):
    if msg[0]['role'] == 'system':
        # we must replace the whole dict here. If we just update the content, 
        # all sample's message will be affected.
        msg[0] = {
            "role": 'system',
            "content": msg[0]['content'] + "\n\n" + system_prompt
        }
    else:
        msg.insert(0, {'role': 'system', 'content': system_prompt})
    return msg