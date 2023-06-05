import "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

const OPEN_AI_API_KEY = import.meta.env.VITE_OPEN_AI_API_KEY;

const img = <ImageData | null>document.getElementById("img");

function generateImageClassificationsContent(classification: { className: string; probability: number }[]) {
    return `
    Image Classifications:
    Group 1:
        - Labels: '${classification?.[0]?.className ?? ""}'
        - Probability Score: '${classification?.[0]?.probability ?? ""}'
    Group 2:
        - Labels: '${classification?.[1]?.className ?? ""}'
        - Probability Score: '${classification?.[1]?.probability ?? ""}'
    Group 3:
        - Labels: '${classification?.[2]?.className ?? ""}'
        - Probability Score: '${classification?.[2]?.probability ?? ""}' 
    `;
}

function getObjectDetectionContent(objectDetection: { class: string; score: number }[]) {
    let content = "";
    objectDetection.forEach((object, index) => {
        content += `
        Prediction Object ${index + 1}:
            - Label: '${object.class ?? ""}'
            - Probability Score: '${object.score ?? ""}'
        `;
    });
    return content;
}

if (img) {
    const modelClassification = await mobilenet.load();
    const modelObjectDetection = await cocoSsd.load();
    const predictionsClassification = await modelClassification.classify(img);
    const predictionsObjectDetection = await modelObjectDetection.detect(img);

    const responseOpenAi = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
            Authorization: `Bearer ${OPEN_AI_API_KEY}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [
                {
                    role: "system",
                    content: `
                    We need a short description of the content in an image, and we have three classification group predictions with sets of labels that represent the content in the image. As an assistant, you'll help us write this description. Each group of labels will have a probability score property that represents the accuracy of the prediction, ranging from 0 (less likely) to 1 (very likely). Additionally, a list of predicted objects detected in the image will be provided, each with a label and precision score. Object detection will be optional and may not always be present.
                    Use the provided information to write a generic description of the image:
                    Step 1: Carefully analyze all labels to summarize the content of the image, prioritizing those with higher probability scores. Keep in mind that some labels may not accurately represent the true image content. Try your best to determine the topic of the image before proceeding with writing your description.
                    Step 2: Write a description that can be used in an Instagram post for the users account. The caption describes the image and captures the essence of the moment. Also, remember that the image was taken from users camera, so you might need to act as a fist person when is needed about the story behind the image or how you captured the moment. This will help the audience connect with the image and understand its significance.    
                    Step 3: Generate hashtags that are relevant to the description and image content. Consider using hashtags that relate to the image and using language that is engaging and descriptive.  
                    Follow the following rules:
                    1. Ensure that the description and hashtags do not exceed the 2200-character limit. This limit is hardcoded for Instagram captions. 
                    2. Avoid using phrases such as "high probability score", "group of labels", "the object detected", and "score" to represent prediction results. 
                    3. Avoid using time-related words such as "today", "yesterday", "last year", etc., since we do not know when the image was captured. 
                    4. Avoid using words such as "Description:" or "Hashtags:" that explicitly indicate the start of the description or hashtags.
                    5. The image description should be descriptive and not contain wording such as "The image is most likely to be a mountain …". Instead, it should be something like "Mountain view on a nice summer day with a reflection in the lake …". Use your own imagination to come up with a nice caption. The three dots '...' in the example indicate that the text should continue.
                    6. It is good to include a personal touch in your writing. For example, you could say "This is an image I took..." or "This scenery was captured by me..." or "I had the opportunity to take a photo of this great view that I visited...”
                    The three dots '...' in the examples indicate that the text should continue.
                    `,
                },
                {
                    role: "user",
                    content: `
                    ${generateImageClassificationsContent(predictionsClassification)}
                    ${getObjectDetectionContent(predictionsObjectDetection)}
                    `,
                },
            ],
        }),
    });
    const resultOpenAi = await responseOpenAi.json();
    document.getElementById("caption")!.innerHTML = resultOpenAi?.choices?.[0]?.message?.content ?? "";
}
