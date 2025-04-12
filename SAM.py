import torch

def SAMLoss(model, image, label, criterion, rho):
    pred = model(image)
    loss = criterion(pred, label)
    loss.backward(retain_graph=True)

    grads = [param.grad.clone() for param in model.parameters() if param.requires_grad]
    wgrads = [torch.norm(param.grad, p=2) for param in model.parameters() if param.requires_grad]
    norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12

    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
          if param is not None:
            param.data = param.data + rho * grad / norm

    sam_pred = model(image)
    sam_loss = criterion(sam_pred, label)

    with torch.no_grad():
      for param, grad in zip(model.parameters(), grads):
        if param is not None:
          param.data = param.data - rho * grad / norm

    return sam_loss