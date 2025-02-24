# pseudocode / python-like

class RelevanceCalculator:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta  = beta

    def compute_utility(self, node):
        """
        Summation of all utility features
        Example approach: sum or weigh each sub-feature
        """
        u_val = 0.0
        
        # 1) Priority
        priority = node.metadata.get("priority_level", "low")
        if priority == "high": 
            u_val += 0.8
        elif priority == "medium":
            u_val += 0.5
        else:
            u_val += 0.2
        
        # 2) Deadline Closeness
        if "deadline" in node.metadata:
            time_left = node.metadata["deadline"] - current_time()
            # scale in [0..1]
            closeness = max(0.0, 1.0 - (time_left / DAYS_7)) 
            u_val += closeness * 0.5
        
        # 3) Persona alignment
        persona = node.metadata.get("persona", "none")
        if persona == "medical":
            if "shift" in node.title.lower():
                u_val += 0.2
        
        # ... (other features)...

        return u_val

    def compute_cost(self, node):
        """
        Summation of all cost features
        """
        c_val = 0.0
        
        # 1) Complexity
        complexity = node.metadata.get("complexity","small")
        if complexity == "large":
            c_val += 0.6
        elif complexity == "medium":
            c_val += 0.3
        
        # 2) Spam
        spam_prob = node.spam_probability
        c_val += spam_prob * 1.0   # e.g. if spam_prob=0.8 => +0.8 cost
        
        # 3) Deferral count
        deferrals = node.metadata.get("deferrals", 0)
        c_val += deferrals * 0.1
        
        # ... (other features)...

        return c_val

    def compute_relevance(self, node):
        u_val = self.compute_utility(node)
        c_val = self.compute_cost(node)
        r_val = self.alpha * u_val - self.beta * c_val
        return r_val

    def update_params(self, d_alpha, d_beta):
        """
        If the user defers many tasks, maybe we raise beta,
        or if they accept tasks easily, we reduce cost weighting, etc.
        """
        self.alpha += d_alpha
        self.beta  += d_beta
