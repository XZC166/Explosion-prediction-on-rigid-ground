import os

with open('main_case_loop.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Constants
content = content.replace("EPOCHS = 100", "EPOCHS = 400")

# 2. Update Model Architecture
old_model = '''class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )'''
new_model = '''class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )'''
content = content.replace(old_model, new_model)

# 3. Update Feature Engineering
old_feat = '''            Z = R / (blast**(1/3)) if blast > 0 else R
            case_X.append([x, y, z, blast, b_height, height, R, Z])'''
new_feat = '''            Z = R / (blast**(1/3)) if blast > 0 else R
            log_Z = np.log(Z + 1e-5)
            inv_Z = 1.0 / (Z + 1e-5)
            case_X.append([x, y, z, blast, b_height, height, R, Z, log_Z, inv_Z])'''
content = content.replace(old_feat, new_feat)

# 4. Update Input Dim
content = content.replace("model = ImprovedMLP(input_dim=8)", "model = ImprovedMLP(input_dim=10)")

# 5. Update Training Loop and Scheduler
old_train = '''    model = ImprovedMLP(input_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    model_save_path = f'model_{run_id}.pth'
    
    print(f"|-- 数据准备完毕：训练集算例 {len(train_cases)}个，总体测点 {len(X_train)}个 --|")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = mape_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    val_loss += mape_loss(model(batch_X), batch_y).item()
            val_loss /= len(test_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)'''

new_train = '''    model = ImprovedMLP(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    best_val_loss = float('inf')
    model_save_path = f'model_{run_id}.pth'
    
    print(f"|-- 数据准备完毕：训练集算例 {len(train_cases)}个，总体测点 {len(X_train)}个 --|")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = mape_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Evaluate validation loss every epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                val_loss += mape_loss(model(batch_X), batch_y).item()
        val_loss /= len(test_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train MAPE: {train_loss:.4f} | Val MAPE: {val_loss:.4f}")'''

content = content.replace(old_train, new_train)

with open('main_case_loop.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Optimization patched successfully.')
