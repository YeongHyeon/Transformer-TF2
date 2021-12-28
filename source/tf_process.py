import os
import numpy as np
import source.utils as utils

def training(agent, dataset, batch_size, epochs):

    print("\n** Training to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0
    loss_best = 1e+12

    savedir = 'results_tr'
    utils.make_dir(path=savedir, refresh=True)
    utils.make_dir(path=os.path.join(savedir, 'reconstruction'), refresh=False)
    utils.make_dir(path=os.path.join(savedir, 'attention'), refresh=False)

    for epoch in range(epochs):

        list_loss = []
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=0)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            list_loss.append(step_dict['losses']['loss_mean'])
            iteration += 1
            if(minibatch['terminate']): break
        list_loss = np.asarray(list_loss)
        loss_tmp = np.average(list_loss)

        print("Epoch [%d / %d] | Loss: %f" %(epoch, epochs, loss_tmp))


        dict_plot = utils.load_dict2pkl("mnist_ex.pkl")
        minibatch = {'x':np.asarray(dict_plot['x']).astype(np.float32), 'y':np.asarray(dict_plot['y']).astype(np.float32)}
        step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=False)

        for idx_batch in range(minibatch['y'].shape[0]):
            savename = "epoch_%06d_%d.png" %(epoch, minibatch['y'][idx_batch])

            list_img = [minibatch['x'][idx_batch][:, :], step_dict['y'][idx_batch][:, :], step_dict['y_hat'][idx_batch][:, :]]
            list_name = ['$X$', '$Y$', '$\hat{Y}$']
            utils.plot_comparison(list_img=list_img, list_name=list_name, cmap='gray', \
                savepath=os.path.join(savedir, 'reconstruction', savename))

            list_img = [step_dict['enc_attn'][idx_batch][:, :], step_dict['dec_attn'][idx_batch][:, :]]
            list_name = ['$Encoder$', '$Decoder$']
            utils.plot_comparison(list_img=list_img, list_name=list_name, cmap='jet', \
                savepath=os.path.join(savedir, 'attention', savename))

        if(loss_best > loss_tmp):
            loss_best = loss_tmp
            agent.save_params(model='model_1_best_loss')
        agent.save_params(model='model_0_finepocch')

def test(agent, dataset):

    savedir = 'results_te'
    utils.make_dir(path=savedir, refresh=True)

    list_model = utils.sorted_list(os.path.join('Checkpoint', 'model*'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    loss_best = 1e+12
    best_dict = {'auroc_name': '', 'auroc': 0}
    for idx_model, path_model in enumerate(list_model):

        print("\n** Test with %s" %(path_model))
        agent.load_params(model=path_model)
        utils.make_dir(path=os.path.join(savedir, path_model), refresh=False)
        utils.make_dir(path=os.path.join(savedir, path_model, 'reconstruction'), refresh=False)
        utils.make_dir(path=os.path.join(savedir, path_model, 'attention'), refresh=False)

        list_loss = []
        idx_save = 0
        while(True):

            minibatch = dataset.next_batch(batch_size=1, ttv=1)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, training=False)

            for idx_batch in range(minibatch['y'].shape[0]):
                savename = "%d_%08d.png" %(minibatch['y'][idx_batch], idx_save)

                list_img = [minibatch['x'][idx_batch][:, :], step_dict['y'][idx_batch][:, :], step_dict['y_hat'][idx_batch][:, :]]
                list_name = ['$X$', '$Y$', '$\hat{Y}$']
                utils.plot_comparison(list_img=list_img, list_name=list_name, cmap='gray', \
                    savepath=os.path.join(savedir, path_model, 'reconstruction', savename))

                list_img = [step_dict['enc_attn'][idx_batch][:, :], step_dict['dec_attn'][idx_batch][:, :]]
                list_name = ['$Encoder$', '$Decoder$']
                utils.plot_comparison(list_img=list_img, list_name=list_name, cmap='jet', \
                    savepath=os.path.join(savedir, path_model, 'attention', savename))
                idx_save += 1

            list_loss.append(step_dict['losses']['loss_mean'])
            if(minibatch['terminate']): break
        list_loss = np.asarray(list_loss)
        loss_tmp = np.average(list_loss)

        if(loss_best > loss_tmp):
            loss_best = loss_tmp
            name_best = path_model
            best_dict = {'name_best': name_best, 'loss': float(loss_tmp)}

    return best_dict, len(list_model)
