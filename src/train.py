import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def train_model(model, train_loader, val_loader, device, epochs, initial_lr):
    # Define loss function & optimizer
    print("Training started!")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)  # L2 Regularization

    # Define the cosine annealing scheduler
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Track losses & accuracy
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} started")
        # Training Phase
        model.train()
        model.to(device)

        train_running_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            #print(f"Processing batch {batch_idx+1}")
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)

            loss = criterion(predictions, labels)
            #print(f"Loss computed for batch {batch_idx+1}")
            loss.backward()
            optimizer.step()
            #print(f"Batch {batch_idx+1} updated!")
            train_running_loss += loss.item() * inputs.shape[0]

            _, predicted = torch.max(predictions, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = train_running_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Phase
        model.eval()
        val_running_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                predictions = model(inputs)
                loss = criterion(predictions, labels)

                val_running_loss += loss.item() * inputs.shape[0]

                _, predicted = torch.max(predictions, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_running_loss / total_val
        scheduler.step(val_loss)  #scheduler step
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        

        # Logging progress
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.3f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"results/model_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1} completed!")

    return train_losses, val_losses, train_accs, val_accs
