import torch
from torchvision import transforms
from torchvision.models import inception_v3
from sklearn.metrics.pairwise import cosine_similarity
from generate import pipe  # ensure 'pipe' is properly imported

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception = inception_v3(pretrained=True, transform_input=False).eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])
def explain_image(prompt, original_image):
    main_tensor = preprocess(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        main_feat = inception(main_tensor)

    words = prompt.split()
    word_scores = []

    # Only keep non-stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]

    for i, word in enumerate(words):
        if word.lower() in stop_words:
            continue

        # Remove the word
        mod_prompt = " ".join([w for j, w in enumerate(words) if j != i])
        alt_image = pipe(mod_prompt).images[0]
        alt_tensor = preprocess(alt_image).unsqueeze(0).to(device)

        with torch.no_grad():
            alt_feat = inception(alt_tensor)

        similarity = cosine_similarity(
            main_feat.cpu().numpy(), alt_feat.cpu().numpy()
        )[0][0]

        impact_score = 1 - similarity
        word_scores.append((word, impact_score))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores
