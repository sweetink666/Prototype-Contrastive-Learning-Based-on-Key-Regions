This code addresses the challenge of few-shot semantic segmentation for dual-source remote sensing imagery. Unlike conventional prototype networks, our approach is specifically designed to handle dual-modal inputs (e.g., PAN and MS). The core framework incorporates both task-aware sampling and cross-modal alignment mechanisms, and proceeds as follows:

Task Construction:
During the episodic training, we construct support and query sets based on feature-driven sampling strategies.

The support set consists of class-representative samples with high intra-class dispersion to improve generalization.

The query set includes a balanced mix of randomly selected and hard samples (i.e., those resembling incorrect class prototypes), making the task more challenging and effective for training.

Superpixel Pooling with Shared Encoding:
To ensure semantic consistency across the two modalities, we apply superpixel pooling guided by a shared superpixel segmentation. This shared encoding ensures that both PAN and MS branches produce aligned superpixel regions with matching counts and spatial semantics.

Key Region Selection via Cross-Modal Similarity:
For each modality, we compute a region-level importance score based on cross-modal cosine similarity. The top 30% most semantically important regions are retained in each branch, and their intersection defines the set of key regions used for alignment.

Prototype-based Contrastive Alignment:
Within the selected key regions, we perform cross-modal prototype contrastive learning.

Taking a query sample from one modality (e.g., MS) as the anchor, we define positive samples from the other modality (e.g., PAN) as:

The same-class prototype,

The most similar and most dissimilar same-class support samples.

Negative samples include:

The most similar different-class prototype, and

The top-2 most similar different-class support samples.

These comparisons are used to compute a margin-based contrastive loss that encourages semantic alignment across modalities.

Attention-Guided Fusion:
Using the previously computed cross-modal similarity scores as attention weights, we fuse the aligned PAN and MS features within the key regions. This attention-guided fusion enhances the complementary information between modalities.

Final Prediction and Loss:
The fused features are passed through the final classification head to generate query predictions. A cross-entropy loss is computed on top of that, and combined with the contrastive loss to obtain the total training objective.
![总流程新](https://github.com/user-attachments/assets/d2cce177-7a0f-4fc9-93f6-5034b3fddb68)
