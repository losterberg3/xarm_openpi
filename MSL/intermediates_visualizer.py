import orbax.checkpoint as ocp
import jax.numpy as jnp
import jax

def distill_scene(checkpoint_tuple, i):
    v_states = checkpoint_tuple[1] 

    states = v_states[i, 0, :, 0, :]

    scene_vector = jnp.mean(states[0:768, :], axis=0)
    return scene_vector


CKPT_DIR = '/tmp/flax_intermediates'
manager = ocp.CheckpointManager(
    CKPT_DIR,
    options=ocp.CheckpointManagerOptions(max_to_keep=100)
)
step0 = manager.restore(5)
# Load Step 1 (New Scene)
step1 = manager.restore(6)
for i in range(18):
    # Extract the Value tensors for image tokens (0:768)
    v0 = step0[1][i, 0, 0:512, 0, :]
    v1 = step1[1][i, 0, 0:512, 0, :]
    # Calculate how different the scenes are (0.0 to 1.0)
    similarity = jnp.mean(jax.vmap(lambda x, y: jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y)))(v0, v1))
    print(f"Layer {i+1} Scene Similarity: {similarity}")

"""
# If you used a single unnamed sow, it may look like:
x = ckpt["('0',)"]    # shape [18, 1, 968, 1, 256]

# Clean it up
x = x.squeeze(axis=3)   # -> [18, 1, 968, 256]

print(x.shape)

layer = 10
tokens = x[layer, 0]   

import numpy as np

norms = np.linalg.norm(tokens, axis=-1)

VISION_TOKENS = slice(0, 196)
vision_tokens = tokens[VISION_TOKENS]  # [196, 256]

energy = np.linalg.norm(vision_tokens, axis=-1)

H = W = int(np.sqrt(len(energy)))  # 14
energy_map = energy.reshape(H, W)

import matplotlib.pyplot as plt

plt.imshow(energy_map, cmap="inferno")
plt.colorbar()
plt.title(f"Layer {layer} – Vision token energy")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
rgb = pca.fit_transform(vision_tokens)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

rgb_map = rgb.reshape(H, W, 3)

plt.imshow(rgb_map)
plt.title(f"Layer {layer} – PCA semantic map")
plt.axis("off")
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, n_init=10)
labels = kmeans.fit_predict(vision_tokens)

seg_map = labels.reshape(H, W)

plt.imshow(seg_map, cmap="tab10")
plt.title(f"Layer {layer} Token clusters")
plt.axis("off")
plt.show()


plt.imshow(rgb_image)
plt.imshow(energy_map, alpha=0.5, cmap="inferno")
plt.show()
"""