import torch.nn.functional as F

def fgsm(Xs, grad, eps=0.1):
    sign_data_grad = grad.sign()
    perturbed_image = Xs + eps*sign_data_grad
    # try normalization instead of clamping
    # perturbed_image = torch.clamp(perturbed_image, 0., 1.)
    return perturbed_image.detach(), sign_data_grad.detach()

def get_adversarial_examples(model, Xs, ys, eps=0.1):
    model.zero_grad()
    # Watch out! Uses original image
    Xs.requires_grad = True
    logits = model(Xs)
    init_pred = logits.max(1, keepdim=True)[1]
    # If the initial prediction is wrong, attack and gradient could be wrong?
    loss = F.cross_entropy(logits, ys)
    loss.backward()
    image_grad = Xs.grad.data
    # mask = torch.nonzero((init_pred != ys).int())
    # image_grad[mask] = torch.zeros(image_grad[mask].shape[1:])
    # print(image_grad.shape)
    Xs_adv, _ = fgsm(Xs, image_grad, eps)
    # Xs_adv = transforms.Normalize(NORM_MEAN, NORM_STD)(Xs_adv)
    return Xs_adv.detach()