import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Discriminator Step:

    Zero the gradients within the discriminator.
    Generate a batch of fake data by sampling noise using the generator.
    Then, calculate the discriminator output on both real and generated data.
    Use the generator output to compute the discriminator loss.
    Apply backward() on the loss output, and optimize the discriminator.

    Generator Step:

    Zero the gradients in the generator.
    Generate a fake data batch by sampling noise.
    Obtain the discriminator output for the fake data batch to compute the generator loss.
    Apply backward() on the loss and optimize the generator.

    Reshape the generated fake image tensor to dimensions (batch_size x input_channels x img_size x img_size).

    Utilize the 'sample_noise' function for random noise sampling and the 'discriminator_loss' and 'generator_loss' functions for their respective loss calculations.
        
    Inputs:
    - D, G: discriminator and generator models
    - D_solver, G_solver: torch.optim Optimizers 
    - discriminator_loss, generator_loss: Loss functions to use
    - show_every: Show generated samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    loss_D, loss_G = [
    for epoch in range(num_epochs):
        
        print('EPOCH: ', (epoch+1))
        loss_d_epoch, loss_g_epoch = [], []
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device)  # normalize
            
            # Discriminator step
            D.zero_grad()  # Zero gradients
            # Generate fake images
            noise = sample_noise(batch_size, noise_size).to(device)
            noise = noise.view(batch_size, 100, 1, 1)
            fake_images = G(noise)
            # Calculate discriminator outputs for real and fake images
            d_real = D(real_images)
            d_fake = D(fake_images.detach()
            # Calculate discriminator loss
            d_error = discriminator_loss(d_real, d_fake)
            # Backpropagation
            d_error.backward()
            D_solver.step()  # Optimize discriminator
    
            # Generator step
            G_solver.zero_grad()  # Zero gradients
            # Calculate discriminator output for the fake images
            d_fake = D(fake_images)
            # Calculate generator loss
            g_error = generator_loss(d_fake)
            # Backpropagation
            g_error.backward()
            G_solver.step()  # Optimize generator

            
            loss_d_epoch.append(d_error.item())
            loss_g_epoch.append(g_error.item())
            # Logging and visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G: {:.4}'.format(iter_count, d_error.item(), g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels != 1)
                plt.show()
                print()
            iter_count += 1
            
        losses_D.append(np.mean(loss_d_epoch))
        losses_G.append(np.mean(loss_g_epoch))

    plt.figure(figsize=(10, 5))
    plt.plot(losses_D, label="Discriminator")
    plt.plot(losses_G, label="Generator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

    iter_count = 0
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            real_images = preprocess_img(x).to(device)  # normalize
            
    #         # Discriminator loss output, generator loss output, and the fake image output
    #         # is stored in these variables for logging and visualization below
    #         d_error = None
    #         g_error = None
    #         fake_images = None
            
    #         ####################################
    #         #          YOUR CODE HERE          #
    #         ####################################
            
            
    #         ##########       END      ##########
            
    #         # Logging and visualization
    #         if (iter_count % show_every == 0):
    #             print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
    #             disp_fake_images = deprocess_img(fake_images.data)  # denormalize
    #             imgs_numpy = (disp_fake_images).cpu().numpy()
    #             show_images(imgs_numpy[0:16], color=input_channels!=1)
    #             plt.show()
    #             print()
    #         iter_count += 1