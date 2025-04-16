import torch

def SAMLoss(model, image, label, criterion, optimizer, rho):
    # 1st forward-backward
    model.eval()
    pred = model(image)
    loss = criterion(pred, label)
    loss.backward()

    grads = [param.grad.clone() for param in model.parameters() if param.requires_grad]
    grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))

    # backup
    backup = [param.data.clone() for param in model.parameters() if param.requires_grad]

    # perturb
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param.add_(rho * grad / (grad_norm + 1e-12))

    # 2nd forward-backward
    model.train()
    model.zero_grad()
    pred_perturbed = model(image)
    loss_perturbed = criterion(pred_perturbed, label)
    loss_perturbed.backward()

    # restore
    with torch.no_grad():
        for param, backup_param in zip(model.parameters(), backup):
            param.data.copy_(backup_param)

    optimizer.step()
    optimizer.zero_grad()

    return loss_perturbed
