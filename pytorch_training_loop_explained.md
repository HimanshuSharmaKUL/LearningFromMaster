
#### 🔁 Training Loop Overview

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(100):
    xb, yb = get_batch('train')         # 1. Get input-target pairs
    logits, loss = m(xb, yb)            # 2. Forward pass (calls m.forward())
    optimizer.zero_grad(set_to_none=True) # 3. Reset old gradients
    loss.backward()                     # 4. Backward pass - compute gradients
    optimizer.step()                    # 5. Use gradients to update weights
    print(loss.item())                  # 6. Show current loss
```

---

#### ✅ `m.parameters()` — What is passed to the optimizer?

This returns a **generator of all the trainable parameters** in the model (`nn.Module`).

Under the hood:
- Every `nn.Module` stores parameters in `self._parameters`.
- Calling `model.parameters()` yields all `nn.Parameter` objects (tensors with `requires_grad=True`).

So, you're telling the optimizer:
> “These are the parameters you need to update during training.”

---

#### ✅ `optimizer.zero_grad(set_to_none=True)` — Why zero gradients?

PyTorch **accumulates gradients by default**.

- Without zeroing, each `.backward()` call would stack on previous gradients.
- This line clears gradients to avoid mixing gradient info between batches.

`set_to_none=True` is slightly more efficient than setting them to zero.

---

#### ✅ `loss.backward()` — What does this do?

This triggers **autograd**, PyTorch's automatic differentiation engine.

- Computes ∂loss/∂param for each trainable parameter.
- It does this by traversing the **computational graph** created during the forward pass.

Each operation (like `matmul`, `add`, etc.) records how to backpropagate. `.backward()` follows these steps to compute gradients and stores them in `.grad`.

---

#### ✅ `optimizer.step()` — What does it do?

This step tells the optimizer:
> “Now that gradients are computed, use them to update parameters.”

For **AdamW**, this includes:
- Momentum-based updates
- Adaptive learning rates
- Weight decay (decoupled from gradient)

Internally, it’s a sophisticated version of:
```python
param = param - lr * gradient
```

---

#### 🔄 How is `loss.backward()` connected to `optimizer.step()`?

1. `loss.backward()` fills `.grad` fields with gradients.
2. `optimizer.step()` reads those `.grad`s and updates parameters.
3. Next iteration, we clear the gradients using `zero_grad()`.

They work together in every training step.

---

#### ❓ Is `.backward()` a method of `loss`?

Yes — because `loss` is a PyTorch `Tensor`.

Even though it’s “just a return value” from the model, it still holds the **history of operations** (via `grad_fn`) that created it.

So when you do:

```python
loss.backward()
```

It:
- Triggers backpropagation
- Computes gradients of loss w.r.t. all model parameters

If you convert loss to a Python float using `loss.item()`, you lose the graph and `.backward()` will no longer work.

---

#### 🧠 Summary Table

| Step | Purpose | What's Happening Under the Hood |
|------|---------|-------------------------------|
| `m.parameters()` | Get model weights | Yields all trainable parameters |
| `optimizer.zero_grad()` | Clear gradients | Prevents gradient accumulation |
| `loss.backward()` | Compute gradients | Uses autograd + chain rule |
| `optimizer.step()` | Update weights | Uses gradients to perform parameter updates |
| `.backward()` on loss | Triggers backprop | Because `loss` is a tensor with grad history |

