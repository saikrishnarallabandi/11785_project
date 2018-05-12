def mixup_data(self,x, y, alpha=1.0, use_cuda=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]

        if use_cuda:
            index = torch.from_numpy(np.random.randint(0,batch_size,(6,))).cuda()
        else:
            index = torch.from_numpy(np.random.randint(0,batch_size,(6,)))
        for i in range(index.size(0)//2):
            segment = np.random.randint(0,x[i].size(0) - 5,(1,))[0]
            #print(x[i + index.size(0)//2].size(), i + index.size(0)//2, segment)
            #print(torch.mean(x[i + index.size(0)//2][segment:segment+5], 0).size())
            x[i] = x[i] +  torch.mean(x[i + index.size(0)//2][segment:segment+5],0 )
        #mixed_x = lam * x + (1 - lam) * x[index, :]
        #y_a, y_b = y, y[index]
        return x


    def mixup_criterion(self,criterion, pred, y_a, y_b, lam, label_mask):
        loss_a = criterion(pred.contiguous().view(-1, len(self.data_loader.vocab)), y_a[:, 1:].contiguous().view(-1))
        loss_b = criterion(pred.contiguous().view(-1, len(self.data_loader.vocab)), y_b[:, 1:].contiguous().view(-1))
        var_label_mask = to_variable(label_mask[:, 1:].contiguous())
        loss_a = (loss_a.view(var_label_mask.size()) * var_label_mask.float()).sum(1).mean()
        loss_b = (loss_b.view(var_label_mask.size()) * var_label_mask.float()).sum(1).mean()
        return lam * loss_a + (1 - lam) * loss_b
