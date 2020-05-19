import torch

class QValues():
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @staticmethod 
    def get_current(policy_net, states, actions):
        return policy_net(states.resize_((32, 113)).float()).gather(dim=1, index=actions.unsqueeze(-1))
    @staticmethod 
    def get_next(target_net, next_states):        
        next_states = next_states.resize_((32, 113)).float()
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=QValues.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_mask] = target_net(non_final_next_states.resize_((32, 113)).float()).max(dim=1)[0].detach()
        return values
