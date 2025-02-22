import os
import math
import openai
from typing import List, Dict, Optional, Any, Tuple

openai.api_key = os.getenv("OPENAI_API_KEY")


# ------------------------------------------------------------------------------------
# A. DATA STRUCTURES: NODES, EDGES, KNOWLEDGE GRAPH
# ------------------------------------------------------------------------------------

class Node:
    """
    Represents a piece of information (task/email/event/doc) in the knowledge graph.

    Key fields:
      - utility (U_i(t))
      - cost (C_i(t))
      - relevance (R_i(t)) = alpha * utility - beta * cost
      - spam_probability (for logistic spam formula)
      - dependencies (parents) is handled by the adjacency in KnowledgeGraph
    """
    def __init__(self, 
                 node_id: str,
                 title: str="",
                 node_type: str="task",
                 completed: bool=False):
        self.node_id = node_id
        self.title = title
        self.node_type = node_type
        self.completed = completed

        # Values used in R_i(t) computations
        self.utility: float = 0.0
        self.cost: float = 0.0
        self.relevance: float = 0.0   # computed

        # For spam classification
        self.spam_probability: float = 0.0

        # Arbitrary metadata (e.g., deadline, domain context, user persona, etc.)
        self.metadata: Dict[str, Any] = {}

class KnowledgeGraph:
    """
    A minimal dynamic knowledge graph:
      - nodes: id -> Node
      - edges: from -> [list of children]
    Dependencies typically from parent to child (D(Ti,Tj)=1).
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = {}  # adjacency: node -> list of child nodes

    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def add_edge(self, from_id: str, to_id: str):
        if from_id not in self.edges:
            self.edges[from_id] = []
        self.edges[from_id].append(to_id)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def all_nodes(self):
        return list(self.nodes.values())

    def get_children(self, node_id: str) -> List[Node]:
        """
        Return children of a given node in the DAG.
        """
        child_ids = self.edges.get(node_id, [])
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_parents(self, node_id: str) -> List[Node]:
        """
        Return parents of node_id by scanning adjacency.
        For large graphs, you'd store reverse edges or do a faster approach.
        """
        parents = []
        for potential_parent, children in self.edges.items():
            if node_id in children:
                parents.append(self.nodes[potential_parent])
        return parents


# ------------------------------------------------------------------------------------
# B. CORE MATH CLASSES
# ------------------------------------------------------------------------------------

class RelevanceCalculator:
    """
    Universal Relevance:
       R_i(t) = alpha * U_i(t) - beta * C_i(t)
    Also includes updates to alpha/beta for user feedback or persona-based weighting.
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def compute_relevance(self, node: Node) -> float:
        return self.alpha * node.utility - self.beta * node.cost

    def update_params(self, d_alpha: float, d_beta: float):
        # e.g. from feedback, or from agent skill weighting
        self.alpha += d_alpha
        self.beta  += d_beta


def logistic_spam_probability(features: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Example logistic function for spam:
      P(spam) = 1 / (1 + e^(- w^T x))
    """
    # dot product
    dot = 0.0
    for f, val in features.items():
        wval = weights.get(f, 0.0)
        dot += wval * val
    return 1.0 / (1.0 + math.exp(-dot))


def critical_path_dag(kg: KnowledgeGraph) -> List[str]:
    """
    Solve for a 'critical path' in the DAG maximizing sum of R_i(t).
    For simplicity, we'll do a topological approach + dynamic programming.

    We'll assume all tasks are 'active' (or skip completed tasks).
    We'll store dp[node_id] = best sum of R_i(t) from node to the end of path.
    Then reconstruct path.

    In a real system, you might factor in durations or other scheduling constraints.
    """
    # 1. get topological order
    topo = topological_sort(kg)
    # 2. dp[node] = node.relevance + max( dp[child] ) over children
    dp = {}
    best_child = {}

    for nid in reversed(topo):
        node = kg.get_node(nid)
        if node is None or node.completed:
            dp[nid] = 0.0
            best_child[nid] = None
            continue
        my_val = node.relevance
        # find best child
        best_score = 0.0
        best_cid   = None
        children   = kg.get_children(nid)
        for c in children:
            c_score = dp.get(c.node_id, 0.0)
            if c_score > best_score:
                best_score = c_score
                best_cid   = c.node_id
        dp[nid] = my_val + best_score
        best_child[nid] = best_cid

    # pick the node with max dp
    best_start = None
    best_sum   = 0.0
    for nid in topo:
        if dp[nid] > best_sum:
            best_sum   = dp[nid]
            best_start = nid

    # reconstruct path
    path = []
    cur = best_start
    while cur is not None:
        path.append(cur)
        cur = best_child[cur]
    return path


def topological_sort(kg: KnowledgeGraph) -> List[str]:
    """
    A standard Kahn's or DFS-based topological sort of the DAG.
    """
    in_degree = {}
    for nid in kg.nodes:
        in_degree[nid] = 0

    for fr, to_list in kg.edges.items():
        for to in to_list:
            in_degree[to] = in_degree.get(to, 0) + 1

    # queue
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo = []

    while queue:
        node_id = queue.pop()
        topo.append(node_id)
        for c in kg.edges.get(node_id, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return topo


def is_task_actionable(kg: KnowledgeGraph, node_id: str) -> bool:
    """
    Dependency activation:
      Task T_j becomes actionable if all parent tasks T_i are completed or deferred.
    """
    parents = kg.get_parents(node_id)
    for p in parents:
        if not p.completed:
            return False
    return True


# ------------------------------------------------------------------------------------
# C. MULTI-AGENT FRAMEWORK
# ------------------------------------------------------------------------------------
class Agent:
    """
    Base class for an agent with an objective function:
      O_k = ∫ sum( phi_k(v_i) * R_i(t) ) dt
    In code, we'll keep it simpler: each agent has a name & a skill weighting function phi_k.
    """
    def __init__(self, name: str):
        self.name = name

    def phi_k(self, node: Node) -> float:
        """
        Skill-specific weighting of node for agent's objective. Subclasses can override.
        """
        return 1.0

    def step(self):
        """
        The main step or routine the agent does. Each agent can implement differently.
        """
        pass


class MultiAgentCoordinator:
    """
    Orchestrates all agents, unifies their objectives if needed, calculates the sum of
      O_k = sum over k of [ sum( phi_k(v_i)*R_i ) ]
    Potentially also resolves conflicts if multiple agents want different outcomes.
    """
    def __init__(self, agents: List[Agent], knowledge_graph: KnowledgeGraph):
        self.agents = agents
        self.kg = knowledge_graph

    def global_objective(self) -> float:
        """
        Sum of all agent objectives over the tasks in the KG:
          sum_{k in agents} sum_{v_i in KG} [ phi_k(v_i) * R_i(t) ]
        This is a simple approach ignoring time integration.
        """
        total = 0.0
        for node in self.kg.all_nodes():
            for agent in self.agents:
                total += agent.phi_k(node) * node.relevance
        return total

    def step_all_agents(self):
        for agent in self.agents:
            agent.step()


# ------------------------------------------------------------------------------------
# D. LLM CLIENT
# ------------------------------------------------------------------------------------
class LLMClient:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name

    def ask(self, prompt: str, max_tokens=256) -> str:
        # In real code: call openai.ChatCompletion
        # We'll pseudo-return
        return f"Mock LLM response to: {prompt[:80]}"


# ------------------------------------------------------------------------------------
# E. 10 AGENTS WITH FULL MATH INTEGRATION
# ------------------------------------------------------------------------------------

# 1. Domain Inference Agent
class DomainInferenceAgent(Agent):
    """
    Takes user email -> guesses domain context
    We unify or store domain context in KG or user profile
    """
    def __init__(self, name: str, llm: LLMClient):
        super().__init__(name)
        self.llm = llm

    def infer_domain(self, user_email: str) -> str:
        prompt = (
            f"We have a user with email domain {user_email}. Guess domain context."
        )
        resp = self.llm.ask(prompt)
        # parse naive
        if "medical" in resp.lower():
            return "medical_context"
        elif "consult" in resp.lower():
            return "consulting_context"
        return "unknown_context"

    def step(self):
        pass  # no ongoing stepping here

# 2. Onboarding Persona Agent
class OnboardingPersonaAgent(Agent):
    """
    1) Takes domain context
    2) Analyzes user emails
    3) Q&A with user to finalize persona
    4) Possibly modifies alpha/beta or sets some user-specific base utility for tasks
    """
    def __init__(self, name: str, llm: LLMClient):
        super().__init__(name)
        self.llm = llm
        self.user_persona = None

    def generate_persona(self, domain_context: str, email_samples: List[str]) -> str:
        # Summarize email patterns with the LLM
        joined = "\n".join(email_samples[:50])
        prompt = (
            f"Domain: {domain_context}\n"
            f"User's last emails:\n{joined}\n"
            "Guess the user's main persona (medical resident, entrepreneur, etc.)"
        )
        return self.llm.ask(prompt)

    def step(self):
        pass


# 3. Email Parsing & Task Extraction Agent
class EmailParsingAgent(Agent):
    """
    LLM-based extraction of tasks from email
    Then we set utility/cost from that classification
    """
    def __init__(self, name: str, llm: LLMClient, kg: KnowledgeGraph):
        super().__init__(name)
        self.llm = llm
        self.kg = kg

    def parse_email_for_tasks(self, email_text: str):
        prompt = f"Extract tasks from: {email_text[:200]}"
        resp = self.llm.ask(prompt)
        # naive parse -> create Node
        # ...
        return [{"title":"Fix bug","priority":"high"}]

    def phi_k(self, node: Node) -> float:
        # This agent might care about extracting tasks:
        # but for multi-agent objective, we'll do 1.0
        return 1.0

    def step(self):
        # in a real system, listens for new emails
        pass


# 4. Spam & Irrelevance Classification Agent
class SpamAgent(Agent):
    """
    Uses logistic function:
      P(spam | email) = 1 / (1 + e^{- w^T x})
    We can unify that with the LLM or user feedback
    Then sets node.spam_probability -> sets utility=0 or cost high if spam.
    """
    def __init__(self, name: str, llm: LLMClient, kg: KnowledgeGraph):
        super().__init__(name)
        self.llm = llm
        self.kg = kg
        self.weights = {"bias": -1.0}  # example

    def classify_spam(self, node: Node):
        # get features from node
        features = {"bias":1.0}  # trivial
        p_spam = logistic_spam_probability(features, self.weights)
        node.spam_probability = p_spam
        # If p_spam > 0.5 => spam => set utility near 0 or cost high
        if p_spam > 0.5:
            node.utility = 0.0
            node.cost = 1.0

    def step(self):
        # check newly added email nodes
        for n in self.kg.all_nodes():
            if n.node_type == "email" and n.spam_probability == 0.0:
                # classify once
                self.classify_spam(n)


# 5. Content Summarization Agent
class SummarizationAgent(Agent):
    def __init__(self, name: str, llm: LLMClient, kg: KnowledgeGraph):
        super().__init__(name)
        self.llm = llm
        self.kg = kg

    def summarize(self, node: Node):
        if "doc_text" in node.metadata:
            doc_text = node.metadata["doc_text"]
            summary = self.llm.ask(f"Summarize: {doc_text[:2000]}")
            node.metadata["summary"] = summary

    def step(self):
        # maybe we look for doc nodes missing a summary
        for n in self.kg.all_nodes():
            if n.node_type == "document" and "summary" not in n.metadata:
                self.summarize(n)


# 6. Content Classification Agent
class ContentClassificationAgent(Agent):
    """
    Places items into categories (receipt, newsletter, doc, etc.)
    Then might set different utility/cost for each category
    """
    def __init__(self, name: str, llm: LLMClient, kg: KnowledgeGraph):
        super().__init__(name)
        self.llm = llm
        self.kg = kg

    def classify_content(self, node: Node):
        if node.node_type not in ["email","document"]:
            return
        text = node.metadata.get("content","")
        prompt = f"Classify text into [receipt, newsletter, doc, etc.]: {text[:200]}"
        cat = self.llm.ask(prompt)
        node.metadata["category"] = cat
        # set utility/cost
        if "receipt" in cat.lower():
            node.utility = 0.2
            node.cost = 0.1
        elif "newsletter" in cat.lower():
            node.utility = 0.1
            node.cost = 0.2
        # else doc => default ?

    def step(self):
        for n in self.kg.all_nodes():
            if n.node_type in ["email","document"] and "category" not in n.metadata:
                self.classify_content(n)


# 7. Task Prioritization & Relevance Agent
class TaskPrioritizationAgent(Agent):
    """
    Applies R_i(t)= alpha * U_i - beta * C_i
    We'll simply recalc for all nodes each step
    """
    def __init__(self, name: str, kg: KnowledgeGraph, relevance_calc: RelevanceCalculator):
        super().__init__(name)
        self.kg = kg
        self.rc = relevance_calc

    def step(self):
        for node in self.kg.all_nodes():
            node.relevance = self.rc.compute_relevance(node)


# 8. Meeting/Calendar Coordination Agent
class CalendarAgent(Agent):
    """
    Coordinates scheduling, references R_i(t).
    If a new high R_i(t) meeting conflicts with a low R_i(t) existing event => ask user to reschedule
    """
    def __init__(self, name:str, kg:KnowledgeGraph, rc:RelevanceCalculator):
        super().__init__(name)
        self.kg = kg
        self.rc = rc

    def step(self):
        # not implementing the entire logic here, but you get the idea
        pass


# 9. User Chat/Command Agent
class ChatCommandAgent(Agent):
    def __init__(self, name:str, llm: LLMClient, kg: KnowledgeGraph, rc: RelevanceCalculator):
        super().__init__(name)
        self.llm = llm
        self.kg = kg
        self.rc = rc

    def interpret_user_input(self, text: str) -> Dict[str,Any]:
        prompt = f"User typed: {text}. parse into json of (intent, etc.)"
        return {"intent":"list_urgent_tasks"}

    def step(self):
        pass


# 10. System Feedback & Learning Agent
class FeedbackLearningAgent(Agent):
    """
    Observes user interactions and updates node utility/cost or global alpha/beta or agent skill weights
    """
    def __init__(self, name:str, rc:RelevanceCalculator, kg: KnowledgeGraph):
        super().__init__(name)
        self.rc = rc
        self.kg = kg

    def process_feedback(self, user_action: str, node_id: str):
        node = self.kg.get_node(node_id)
        if not node:
            return
        if user_action == "defer" and "marketing" in node.title.lower():
            node.utility -= 0.1
        elif user_action == "not_spam":
            node.utility = max(node.utility, 0.2)
        node.relevance = self.rc.compute_relevance(node)

    def step(self):
        pass


# ------------------------------------------------------------------------------------
# F. FULL IMPLEMENTATION EXTRAS: Dependency Activation, Critical Path, etc.
# ------------------------------------------------------------------------------------

def update_task_actionability(kg: KnowledgeGraph):
    """
    For each task in KG, check if all parents are completed => if so, it's active
    If not active, we might set cost high or something.
    """
    for node in kg.all_nodes():
        if node.node_type == "task":
            active = is_task_actionable(kg, node.node_id)
            if not active and not node.completed:
                # e.g. set cost high
                node.cost += 0.5


def compute_critical_path(kg: KnowledgeGraph):
    """
    Demonstrates the global flow: 
    pick the path that yields the max sum of R_i(t).
    """
    path = critical_path_dag(kg)
    return path


# ------------------------------------------------------------------------------------
# G. EXAMPLE MAIN: Orchestrating Everything
# ------------------------------------------------------------------------------------
def main():
    # SETUP
    kg = KnowledgeGraph()
    rc = RelevanceCalculator(alpha=1.0, beta=1.0)
    llm = LLMClient(model_name="gpt-4")

    # CREATE AGENTS
    domain_agent   = DomainInferenceAgent("DomainAgent", llm)
    persona_agent  = OnboardingPersonaAgent("PersonaAgent", llm)
    parse_agent    = EmailParsingAgent("EmailParser", llm, kg)
    spam_agent     = SpamAgent("SpamAgent", llm, kg)
    summary_agent  = SummarizationAgent("SummAgent", llm, kg)
    classify_agent = ContentClassificationAgent("ClassifyAgent", llm, kg)
    task_prior     = TaskPrioritizationAgent("TaskPrior", kg, rc)
    cal_agent      = CalendarAgent("CalAgent", kg, rc)
    chat_agent     = ChatCommandAgent("ChatAgent", llm, kg, rc)
    feedback_agent = FeedbackLearningAgent("FeedbackAgent", rc, kg)

    # MAKE A MULTI-AGENT COORDINATOR
    coordinator = MultiAgentCoordinator(
        [domain_agent, persona_agent, parse_agent, spam_agent,
         summary_agent, classify_agent, task_prior, cal_agent, chat_agent, feedback_agent],
        kg
    )

    # 1) Domain Inference
    user_domain = domain_agent.infer_domain("dr.jane@harvardmed.edu")
    print(f"**Inferred domain**: {user_domain}")

    # 2) Persona from domain + email samples
    sample_emails = ["Re: Next shift schedule Monday 7am?", "Clinical trial updates..."]
    persona = persona_agent.generate_persona(domain_context=user_domain, email_samples=sample_emails)
    print(f"**Persona**: {persona}")

    # 3) Create email node -> parse tasks -> spam check -> classification
    email_node = Node(node_id="email_1", title="Email about billing bug", node_type="email")
    email_node.metadata["content"] = "Can you fix the billing bug by Friday? Also CC finance."

    kg.add_node(email_node)

    # Agents step
    coordinator.step_all_agents()  # spam agent will check, classification agent might classify, etc.

    # Now parse tasks from that email
    tasks_info = parse_agent.parse_email_for_tasks(email_node.metadata["content"])
    for tinfo in tasks_info:
        tid = "task_"+ tinfo["title"].replace(" ","_")
        new_node = Node(node_id=tid, title=tinfo["title"], node_type="task")
        # if priority=high => utility=0.8
        if tinfo["priority"] == "high":
            new_node.utility = 0.8
        kg.add_node(new_node)
        # link email -> task if you want
        kg.add_edge(email_node.node_id, new_node.node_id)

    # Another step to recalc
    coordinator.step_all_agents()

    # 4) Update Task Actionability
    update_task_actionability(kg)

    # 5) Task prioritization & see critical path
    #    step again so T_i(t) are fresh
    coordinator.step_all_agents()

    # see top tasks
    tasks = [n for n in kg.all_nodes() if n.node_type=="task"]
    tasks.sort(key=lambda x:x.relevance, reverse=True)
    print("**Top tasks by R_i(t)**:")
    for t in tasks:
        print(f"{t.node_id} => R={t.relevance:.2f}, U={t.utility}, C={t.cost}")

    # 6) Compute critical path
    cpath = compute_critical_path(kg)
    print("**Critical Path** (max sum of R_i(t)):", cpath)

    # 7) Feedback scenario: user defers "Fix_bug"
    fix_node = kg.get_node("task_Fix_bug")
    if fix_node:
        feedback_agent.process_feedback("defer", fix_node.node_id)

    # step again to see new R
    coordinator.step_all_agents()

    # 8) see final multi-agent objective
    total_obj = coordinator.global_objective()
    print(f"**Global Multi-Agent Objective**: {total_obj:.2f}")

    print("Done.")


if __name__ == "__main__":
    main()
