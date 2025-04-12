import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def inconsistencyLoss(model, model_prime, image, pred, label, criterion, k = 1, beta = 1.0, rho=0.1, eval_mode = True):
    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    seed = random.randint(0, 2**32 - 1)

    # Weight Initialization
    model_prime.load_state_dict(model.state_dict())
    noise_dict = {}
    with torch.no_grad():
      for name, param in model_prime.named_parameters():
        # noise = 0.001 * torch.normal(0, 1, size=param.data.shape, device=device)
        noise = torch.normal(0, 1, size=param.data.shape, device=device) / math.sqrt(sum(p.numel() for p in model.parameters() if p.requires_grad))
        param.data = param.data + noise #0.1 수치가 애매함 -> normalization (gradient 없애려고 하는건데 0.1은 약간 클수도)
        noise_dict[name] = noise

    # optimizer_prime = optim.SGD(model_prime.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if eval_mode:
        # Without BatchNorm
        model.eval()
        model_prime.eval()
        pred = model(image)
        model.train()

        # Copy BatchNorm running stats
        for m1, m2 in zip(model.modules(), model_prime.modules()):
            if isinstance(m1, nn.BatchNorm2d) and isinstance(m2, nn.BatchNorm2d):
                m2.running_mean.data = m1.running_mean.data.clone()
                m2.running_var.data = m1.running_var.data.clone()
                m2.num_batches_tracked.data = m1.num_batches_tracked.data.clone()

        pred_soft = F.softmax(pred.detach(), dim=1)
        pred_soft = torch.clamp(pred_soft, min=1e-6, max=1.0)
        pred_soft = pred_soft / pred_soft.sum(dim=1, keepdim=True)
    else:
        model_prime.train()

        # torch.manual_seed(seed)
        # pred = model(image)

        
    # Gradient ascent
    for _ in range(k):
        # optimizer_prime.zero_grad()
        with torch.enable_grad():
            loss_kl = 0.0
            if eval_mode:
                loss_kl = -1 * criterion_kl(F.log_softmax(model_prime(image), dim=1), pred_soft)
            else:
                # torch.manual_seed(seed)
                loss_kl = -1 * criterion_kl(F.log_softmax(model_prime(image), dim=1), F.softmax(pred, dim=1))
        loss_kl.backward(retain_graph=True)
        # optimizer_prime.step()

        # SAM-like gradient ascent
        grads = [param.grad.clone() for param in model_prime.parameters() if param.requires_grad]
        wgrads = [torch.norm(param.grad, p=2) for param in model_prime.parameters() if param.requires_grad]
        norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12
        
        # ASAM-like
        # scaled_grads = [grad * (torch.abs(param) + 1e-3) for param, grad in zip(model_prime.parameters(), grads)]
        # norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in scaled_grads]), p=2) + 1e-12

        with torch.no_grad():
            for name_param, grad in zip(model_prime.named_parameters(), grads):
            # for name_param in model_prime.named_parameters():
                name, param = name_param
                if param is not None:
                    # ASAM-like ascent
                    # grad = grad * (torch.abs(param) + 1e-3)
                    # param.data = param.data - noise_dict[name] + rho * grad / norm

                    #SAM-like ascent
                    param.data = param.data - noise_dict[name] + rho * grad / norm

                    # Random ascent
                    # vec = torch.randn_like(param)
                    # param.data = param.data - noise_dict[name] + rho * vec / (torch.norm(vec) + 1e-12)

    if eval_mode:  # model도 eval() 켜기
        return beta * criterion_kl(F.log_softmax(model_prime(image), dim=1), pred_soft)
    else:
        # torch.manual_seed(seed)
        return beta * criterion_kl(F.log_softmax(model_prime(image), dim=1), F.softmax(pred, dim=1))