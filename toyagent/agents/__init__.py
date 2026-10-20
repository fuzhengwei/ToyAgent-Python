"""场景注册表 —— 与 Java Application 的注册顺序一致（教学叙事顺序）：

step01,02,03,04,05,06,09,10,07,12,15,16,18,08,11,17,19,20,21,22,23,25,13,24,14,26
"""
from ..llm import models
from .step01_chat_agent import ChatAgent
from .step02_prompt_agent import PromptAgent
from .step03_react_agent import ReActAgent
from .step04_tool_call_agent import ToolCallAgent
from .step05_memory_agent import MemoryAgent
from .step06_router_agent import RouterAgent
from .step07_rag_agent import RagAgent
from .step08_wiki_agent import WikiAgent
from .step09_mcp_agent import McpAgent
from .step10_skill_agent import SkillAgent
from .step11_registry_agent import RegistryAgent
from .step12_multi_agent import MultiAgent
from .step13_subagent_agent import SubagentAgent
from .step14_a2a_agent import A2AAgent
from .step15_loop_agent import LoopAgent
from .step16_workflow_agent import WorkflowAgent
from .step17_runtime_agent import RuntimeAgent
from .step18_full_agent import FullAgent
from .step19_ask_agent import AskAgent
from .step20_approval_agent import ApprovalAgent
from .step21_sandbox_agent import SandboxAgent
from .step22_event_sourced_agent import EventSourcedAgent
from .step23_plugin_agent import PluginAgent
from .step24_hooks_agent import HooksAgent
from .step25_cli_agent import CliAgent
from .step26_schedule_agent import ScheduleAgent

# 注册顺序即教学叙事顺序（与 Java Application.rebuildAgents 一致）。
SCENARIO_ORDER = [
    "step01", "step02", "step03", "step04", "step05", "step06",
    "step09", "step10", "step07", "step12", "step15", "step16",
    "step18", "step08", "step11", "step17", "step19", "step20",
    "step21", "step22", "step23", "step25", "step13", "step24",
    "step14", "step26",
]

_SCENARIO_CLASSES = {
    "step01": ChatAgent, "step02": PromptAgent, "step03": ReActAgent,
    "step04": ToolCallAgent, "step05": MemoryAgent, "step06": RouterAgent,
    "step07": RagAgent, "step08": WikiAgent, "step09": McpAgent,
    "step10": SkillAgent, "step11": RegistryAgent, "step12": MultiAgent,
    "step13": SubagentAgent, "step14": A2AAgent, "step15": LoopAgent,
    "step16": WorkflowAgent, "step17": RuntimeAgent, "step18": FullAgent,
    "step19": AskAgent, "step20": ApprovalAgent, "step21": SandboxAgent,
    "step22": EventSourcedAgent, "step23": PluginAgent, "step24": HooksAgent,
    "step25": CliAgent, "step26": ScheduleAgent,
}


def create_agent(step_id: str):
    """按当前配置装配场景：真实模型优先，无 Key 则装配 Mock 模型。"""
    real = models.open_ai_or_null()
    model = real if real is not None else models.mock(step_id)
    return _SCENARIO_CLASSES[step_id](model)
