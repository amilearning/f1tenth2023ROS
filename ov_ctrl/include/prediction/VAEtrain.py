import torch 
from tqdm import tqdm
from matplotlib import pyplot as plt

def VAEtrain(args, model, train_loader, test_loader, writer):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    ## interation setup
    epochs = tqdm(range(args['max_iter'] // len(train_loader) + 1))

    ## training
    count = 0
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        for i, batch_data in train_iterator:

            if count > args['max_iter']:
                return model
            count += 1

            train_data = batch_data[:,:,0:7].to(args['device'])

            ## reshape
            batch_size = train_data.size(0)
            # example_size = past_data.size(1)
            # image_size = past_data.size(1), past_data.size(2)
            # past_data = (
            #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
            # )
            # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

            mloss, recon_x, info, theta_mean_, theta_logvar_ = model(train_data)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(mloss.mean())})
        writer.add_scalar("train_loss", float(mloss.mean()), epoch)

        model.eval()
        eval_loss = 0
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )

        with torch.no_grad():
            for i, batch_data in test_iterator:
                test_data = batch_data[:,:,0:7].to(args['device'])

                ## reshape
                batch_size = test_data.size(0)
                # example_size = past_data.size(1)
                # past_data = (
                #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
                # )
                # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

                mloss, recon_x, info, theta_mean_, theta_logvar_ = model(test_data)

                eval_loss += mloss.mean().item()

                test_iterator.set_postfix({"eval_loss": float(mloss.mean())})

                # if i == 0:
                #     for idx in range(len(recon_x)):
                #         plt.plot(recon_x[idx,:,0].cpu(), recon_x[idx,:,0].cpu(), 'r.')
                #         plt.plot(past_data[idx,:,0].cpu(), past_data[idx,:,0].cpu(), 'g.')                        
                #         plt.pause(0.01)
                #     plt.clf()
                    # nhw_orig = past_data[0].view(example_size, image_size[0], -1)
                    # nhw_recon = recon_x[0].view(example_size, image_size[0], -1)                    
                    # writer.add_images(f"original{i}", nchw_orig, epoch)
                    # writer.add_images(f"reconstructed{i}", nchw_recon, epoch)

        eval_loss = eval_loss / len(test_loader)
        writer.add_scalar("eval_loss", float(eval_loss), epoch)
        print("Evaluation Score : [{}]".format(eval_loss))

    return model
