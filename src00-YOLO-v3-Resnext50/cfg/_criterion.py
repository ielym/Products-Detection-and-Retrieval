import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).float().cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input.float(), F.one_hot(target, num_classes=input.size(-1)).float(), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


class Bi_Tempered_Logistic_Loss(nn.Module):

    def __init__(self, label_smoothing=0.1, t1=0.2, t2=1.2, num_iters=5, reduction='mean'):
        '''
            t1: Temperature 1 (< 1.0 for boundedness).
            t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
            label_smoothing: Label smoothing parameter between [0, 1).
            num_iters: Number of iterations to run the method.
        '''
        super(Bi_Tempered_Logistic_Loss, self).__init__()
        self.label_smoothing = label_smoothing
        self.t1 = t1
        self.t2 = t2
        self.num_iters = num_iters
        self.reduction = reduction


    def log_t(self, u, t):
        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    def exp_t(self, u, t):
        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    def compute_normalization_fixed_point(self, activations, t):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < self.num_iters:
            i += 1
            logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)

        return -self.log_t(1.0 / logt_partition, t) + mu

    def compute_normalization(self, activations, t):
        """Returns the normalization value for each example.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """
        if t < 1.0:
            return None  # not implemented as these values do not occur in the authors experiments...
        else:
            return self.compute_normalization_fixed_point(activations, t)

    def tempered_softmax(self, activations, t):
        """Tempered softmax function.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """
        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
        else:
            normalization_constants = self.compute_normalization(activations, t)

        return self.exp_t(activations - normalization_constants, t)

    def forward(self, inputs, targets):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        labels: A tensor with shape and dtype as activations.
        Returns:
        A loss tensor.
        """
        targets = inputs.data.clone().zero_().scatter_(1, targets.unsqueeze(1), 1)
        if self.label_smoothing > 0.0:
            num_classes = targets.shape[-1]
            targets = (1 - num_classes / (num_classes - 1) * self.label_smoothing) * targets + self.label_smoothing / (num_classes - 1)

        probabilities = self.tempered_softmax(inputs, self.t2)

        temp1 = (self.log_t(targets + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * targets
        temp2 = (1 / (2 - self.t1)) * (torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss_values = temp1 - temp2

        if self.reduction == 'mean':
            loss = torch.mean(torch.sum(loss_values, dim=-1), dim=-1)
        else:
            loss = torch.sum(torch.sum(loss_values, dim=-1), dim=-1)

        return loss


class Weighted_Bi_Tempered_Logistic_Loss(nn.Module):

    def __init__(self, balance:list, label_smoothing=0.1, t1=0.2, t2=1.2, num_iters=5, reduction='mean'):
        '''
            t1: Temperature 1 (< 1.0 for boundedness).
            t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
            label_smoothing: Label smoothing parameter between [0, 1).
            num_iters: Number of iterations to run the method.
        '''
        super(Weighted_Bi_Tempered_Logistic_Loss, self).__init__()
        self.balance = balance
        self.label_smoothing = label_smoothing
        self.t1 = t1
        self.t2 = t2
        self.num_iters = num_iters
        self.reduction = reduction


    def log_t(self, u, t):
        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    def exp_t(self, u, t):
        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    def compute_normalization_fixed_point(self, activations, t):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < self.num_iters:
            i += 1
            logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)

        return -self.log_t(1.0 / logt_partition, t) + mu

    def compute_normalization(self, activations, t):
        """Returns the normalization value for each example.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """
        if t < 1.0:
            return None  # not implemented as these values do not occur in the authors experiments...
        else:
            return self.compute_normalization_fixed_point(activations, t)

    def tempered_softmax(self, activations, t):
        """Tempered softmax function.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """
        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
        else:
            normalization_constants = self.compute_normalization(activations, t)

        return self.exp_t(activations - normalization_constants, t)

    def forward(self, inputs, targets):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        labels: A tensor with shape and dtype as activations.
        Returns:
        A loss tensor.
        """
        balance_mask = targets.data.clone().zero_()

        for i in range(len(self.balance)):
            idxs = targets.data.cpu() == i
            balance_mask[idxs] = self.balance[i]

        targets = inputs.data.clone().zero_().scatter_(1, targets.unsqueeze(1), 1)

        if self.label_smoothing > 0.0:
            num_classes = targets.shape[-1]
            targets = (1 - num_classes / (num_classes - 1) * self.label_smoothing) * targets + self.label_smoothing / (num_classes - 1)

        probabilities = self.tempered_softmax(inputs, self.t2)

        temp1 = (self.log_t(targets + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * targets
        temp2 = (1 / (2 - self.t1)) * (torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss_values = temp1 - temp2

        if self.reduction == 'mean':
            loss = torch.mean(torch.sum(loss_values, dim=-1) * balance_mask.float(), dim=-1)
        else:
            loss = torch.sum(torch.sum(loss_values, dim=-1) * balance_mask.float(), dim=-1)

        return loss

class Smoolth_L1_Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Smoolth_L1_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, selected_pns):
        selected_pns = selected_pns.unsqueeze(dim=1)
        t = torch.abs(inputs - targets) * selected_pns
        ret = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret