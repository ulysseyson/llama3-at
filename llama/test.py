import torch


class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.attention = {}

    def get_attention_scores(self, module, input, output):
        attention_scores = (
            module.attention.scores.to(torch.float32).detach().cpu().numpy()
        )
        log_dict = {}
        log_dict["step"] = (
            0 if len(self.attention) == 0 else self.attention["step"][-1] + 1
        )
        for head in range(attention_scores.shape[0]):
            log_dict[f"head_{head}"] = attention_scores[head][0].tolist()
        for key in log_dict:
            if self.attention.get(key) is None:
                self.attention[key] = [log_dict[key]]
            else:
                self.attention[key].append(log_dict[key])

    # return get_attention_scores

    def evaluate(self, data):
        return self.model(data)

    def register_hook(self, layer_num):
        layer = self.model.layers[layer_num]
        layer.register_forward_hook(self.get_attention_scores)

    def reset(self):
        self.attention = {}
