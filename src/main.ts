import "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { pipeline } from "@xenova/transformers";

// const imageToText = await pipeline("image-to-text", "Xenova/vit-gpt2-image-captioning");
// const output = await imageToText("./src/assets/img/image-5.jpg");
// console.log(output);

// Examples:
// https://source.unsplash.com/rFGkDWGk85c
// https://source.unsplash.com/qMs3Ui0bcB4
// https://source.unsplash.com/KZdHrGgldZU
// https://source.unsplash.com/04X1Yp9hNH8
// https://source.unsplash.com/M5t0IHerKw8
// https://source.unsplash.com/zIQMg0a1TvE

// const imageToText = await pipeline("image-to-text", "Xenova/vit-gpt2-image-captioning");
// const output = await imageToText("https://source.unsplash.com/zIQMg0a1TvE");
// console.log(output);

// const classifier = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32");
// const imageUrl = "./src/assets/img/image-5.jpg";
// const labelsForClassification = [
//     "Deep within the ancient woods, a red fox darted through the tangled undergrowth, navigating the labyrinth of towering trees.",
//     "As I ventured into the dark woods, a sense of unease washed over me, heightened by the sight of a mysterious red glow emanating from a hidden labyrinth.",
//     "The vibrant hues of autumn painted the woods in a tapestry of red, guiding me through nature's labyrinth with a mesmerizing beauty.",
//     "Lost in the dense woods, I stumbled upon a peculiar red door, leading me deeper into a labyrinth of secrets and enigmatic passages.",
//     "The silence of the moonlit woods was broken only by the flutter of a scarlet bird, gracefully flying through the labyrinth of branches.",
//     "A solitary red rose stood amidst the labyrinthine thicket of the ancient woods, its beauty a stark contrast to the surrounding darkness.",
//     "The haunting whispers of the woods echoed through the labyrinth of my mind, their eerie voices intertwined with the vivid red imagery.",
//     "With each step, the labyrinth of trees in the woods seemed to shift, leaving me disoriented and enveloped in a maze of crimson uncertainty.",
//     "A burst of red mushrooms sprouted along the mossy path, leading me further into the heart of the enchanted woods' intricate labyrinth.",
//     "In the depths of the enchanted woods, I stumbled upon a forgotten red book, its pages unraveling a tale of a cursed labyrinth waiting to be conquered.",
// ];
// const scores = await classifier(imageUrl, labelsForClassification);
// console.log(scores);

const OPEN_AI_API_KEY = import.meta.env.VITE_OPEN_AI_API_KEY;

let fileName = "";
let filePath = "";

document.getElementById("file")!.addEventListener("change", async (event) => {
    const eventTarget = event.target as HTMLInputElement;
    if (!eventTarget.files) {
        return;
    }
    const attachedFile = eventTarget.files[0];
    let fileNameTrim = attachedFile.name.split(".");

    fileNameTrim.pop();
    fileName = fileNameTrim.join(".");
    reader.readAsDataURL(attachedFile);
    document.getElementById("caption")!.innerHTML = "";
    filePath = URL.createObjectURL(eventTarget.files[0]);
});

document.getElementById("generate")!.addEventListener("click", () => {
    generateInstagramCapture();
});

async function generateInstagramCapture() {
    const imageToText = await pipeline("image-to-text", "Xenova/vit-gpt2-image-captioning");
    const imageCapture = await imageToText(filePath);

    const responseImageCaptureWords = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
            Authorization: `Bearer ${OPEN_AI_API_KEY}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [
                {
                    role: "user",
                    content: `Extract only the nouns and verbs from the following text: "${imageCapture[0].generated_text}". Output them as a single array of strings in JavaScript format, such as ["noun1", "noun2", "verb1", "verb2"]. Only one array should be returned.`,
                },
            ],
        }),
    });

    const responseImageCaptureWordsJson = await responseImageCaptureWords.json();

    console.log(responseImageCaptureWordsJson.choices[0].message.content);

    const indexOfFirstBracketOfImageCaptureWords =
        responseImageCaptureWordsJson.choices[0].message.content.indexOf("[");
    const indexOfLastBracketsOfImageCaptureWords =
        responseImageCaptureWordsJson.choices[0].message.content.lastIndexOf("]") + 1;
    const onlyArrayImageCaptureWords = responseImageCaptureWordsJson.choices[0].message.content.slice(
        indexOfFirstBracketOfImageCaptureWords,
        indexOfLastBracketsOfImageCaptureWords
    );

    const arrayPredictedClassesXenova = JSON.parse(onlyArrayImageCaptureWords);
    const arrayPredictedClassesTensorflow = [] as any[];
    const arrayPredictedObjectsTensorflow = [] as any[];

    const img = <ImageData | null>document.getElementById("image");

    if (img) {
        const modelClassification = await mobilenet.load();
        const modelObjectDetection = await cocoSsd.load();
        const predictionsClassification = await modelClassification.classify(img);
        const predictionsObjectDetection = await modelObjectDetection.detect(img);

        predictionsClassification.forEach((prediction) => {
            arrayPredictedClassesTensorflow.push(...prediction.className.split(",").flat());
        });

        predictionsObjectDetection.forEach((prediction) => {
            arrayPredictedObjectsTensorflow.push(...prediction.class.split(",").flat());
        });
    }

    const arrayPredictedClasses = [
        ...new Set([
            ...arrayPredictedClassesXenova,
            ...arrayPredictedClassesTensorflow,
            ...arrayPredictedObjectsTensorflow,
        ]),
    ];

    const classifier = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32");
    const scoresPredictions = await classifier(filePath, arrayPredictedClasses);

    const scoresPredictionsSorted = scoresPredictions.sort((a: any, b: any) => b.score - a.score);

    const scoresPredictionsSortedTop5 = scoresPredictionsSorted.slice(0, 5);

    const scoresPredictionsSortedTop5String = scoresPredictionsSortedTop5
        .map((prediction: any) => prediction.label)
        .join(", ");

    // const responseImageText = await fetch("https://api.openai.com/v1/chat/completions", {
    //     method: "POST",
    //     headers: {
    //         Authorization: `Bearer ${OPEN_AI_API_KEY}`,
    //         "Content-Type": "application/json",
    //     },
    //     body: JSON.stringify({
    //         model: "gpt-3.5-turbo",
    //         messages: [
    //             {
    //                 role: "user",
    //                 content: `First, write 10 sentences using the following five words: "${scoresPredictionsSortedTop5String}". The sentences must not mention any specific location, such as the name of a country or city. Then, output a JavaScript array containing the sentences in the following format: ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5", "sentence6", "sentence7", "sentence8", "sentence9", "sentence10"]. Return only one array of strings.`,
    //             },
    //         ],
    //     }),
    // });
    // const responseImageTextJson = await responseImageText.json();

    // const indexOfFirstBracketOfImageText = responseImageTextJson.choices[0].message.content.indexOf("[");
    // const indexOfLastBracketsOfImageText = responseImageTextJson.choices[0].message.content.lastIndexOf("]") + 1;
    // const onlyArrayImageText = responseImageTextJson.choices[0].message.content.slice(
    //     indexOfFirstBracketOfImageText,
    //     indexOfLastBracketsOfImageText
    // );

    // const arrayImageTexts = JSON.parse(onlyArrayImageText);

    // const scoresTextDescriptions = await classifier(filePath, arrayImageTexts);

    // const [scoresTextDescriptionsSorted] = scoresTextDescriptions.sort((a: any, b: any) => b.score - a.score);
    // console.log(scoresTextDescriptionsSorted.label);

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
                    We need a short description of the content in an image, and we have five labels that represent the content in the image. As an assistant, your role is to help us write this description. In addition to having all labels related to the content of the image, you will also be provided with the title of the image. The title might mention the location where the photograph was taken, as well as describe what is in the image.
                    Use the provided information to write a generic description of the image:
                    Step 1: Carefully analyze all labels to summarize the content of the image. Keep in mind that some labels may not accurately represent the true image content. Try your best to determine the topic of the image before proceeding with writing your description.
                    Step 2: Carefully analyze the title of the image. Extract as much information as possible about the place or possible location, and use your broad knowledge to come up with a nice summary. Don't be shy to use your own research to find more details about the place and location.  
                    Step 3: Write a description that can be used in an Instagram post for the users account. The caption describes the image and captures the essence of the moment. Also, remember that the image was taken from users camera, so you might need to act as a fist person when is needed about the story behind the image or how you captured the moment. This will help the audience connect with the image and understand its significance.
                    Step 4: Generate hashtags that are relevant to the description and image content. Consider using hashtags that relate to the image, and use engaging and descriptive language. Also, try to generate as many hashtags as possible related to the location, tourist attractions, or parts of the image shown. Hashtags are very important to engage with the audience.
                    You must follow the following rules:
                    1. Summarize the capture in a single sentence and ensure that the description and hashtags do not exceed the 2200-character limit. This limit is hardcoded for Instagram captions. 
                    2. Do not use using time-related words such as "today", "yesterday", "last year", etc., since we do not know when the image was captured. 
                    3. Do not use using words such as "Description:" or "Hashtags:" that explicitly indicate the start of the description or hashtags.
                    4. The image description should be descriptive and not contain wording such as "The image is most likely to be a mountain …". Instead, it should be something like "Mountain view on a nice summer day with a reflection in the lake …". Use your own imagination to come up with a nice caption. The three dots '...' in the examples indicate that the text should continue.
                    5. It is good to include a personal touch in your writing. For example, you could say "This is an image I took..." or "This scenery was captured by me..." or "I had the opportunity to take a photo of this great view that I visited...”
                    `,
                },
                {
                    role: "user",
                    content: `
                        Image Labels: "${scoresPredictionsSortedTop5String}"
                        Image Title: "${fileName}"
                        `,
                },
            ],
        }),
    });
    const resultOpenAi = await responseOpenAi.json();
    document.getElementById("caption")!.innerHTML = resultOpenAi?.choices?.[0]?.message?.content ?? "";
}

// document.getElementById("generate")!.addEventListener("click", () => {
//     generateInstagramCapture();
// });

const reader = new FileReader();
reader.onload = (event) => {
    if (event.target) {
        document.getElementById("image")!.setAttribute("src", event.target.result as string);
    }
};

// function generateImageClassificationsContent(classification: { className: string; probability: number }[]) {
//     return `
//     Image Classifications:
//     Group 1:
//         - Labels: '${classification?.[0]?.className ?? ""}'
//         - Probability Score: '${classification?.[0]?.probability ?? ""}'
//     Group 2:
//         - Labels: '${classification?.[1]?.className ?? ""}'
//         - Probability Score: '${classification?.[1]?.probability ?? ""}'
//     Group 3:
//         - Labels: '${classification?.[2]?.className ?? ""}'
//         - Probability Score: '${classification?.[2]?.probability ?? ""}'
//     `;
// }

// function getObjectDetectionContent(objectDetection: { class: string; score: number }[]) {
//     let content = "";
//     objectDetection.forEach((object, index) => {
//         content += `
//         Prediction Object ${index + 1}:
//             - Label: '${object.class ?? ""}'
//             - Probability Score: '${object.score ?? ""}'
//         `;
//     });
//     return content;
// }

// function generateTitleContent(title: string) {
//     return `
//     Title of the image: '${title ?? ""}'
//     `;
// }

// async function generateInstagramCapture() {
//     const img = <ImageData | null>document.getElementById("image");
//     if (img) {
//         const modelClassification = await mobilenet.load();
//         const modelObjectDetection = await cocoSsd.load();
//         const predictionsClassification = await modelClassification.classify(img);
//         const predictionsObjectDetection = await modelObjectDetection.detect(img);

//         const responseOpenAi = await fetch("https://api.openai.com/v1/chat/completions", {
//             method: "POST",
//             headers: {
//                 Authorization: `Bearer ${OPEN_AI_API_KEY}`,
//                 "Content-Type": "application/json",
//             },
//             body: JSON.stringify({
//                 model: "gpt-3.5-turbo",
//                 messages: [
//                     {
//                         role: "system",
//                         content: `
//                         We need a short description of the content in an image, and we have three classification group predictions with sets of labels that represent the content in the image. As an assistant, your role is to help us write this description. Each group of labels will have a probability score property that represents the accuracy of the prediction, ranging from 0 (less likely) to 1 (very likely). Additionally, a list of predicted objects detected in the image will be provided, each with a label and precision score. Object detection will be optional and may not always be present. In addition to having all the information related to the content of the image, you will also be provided with the title of the image. The title might mention the location where the photograph was taken, as well as describe what is in the image.
//                         Use the provided information to write a generic description of the image:
//                         Step 1: Carefully analyze all labels to summarize the content of the image, prioritizing those with higher probability scores. Keep in mind that some labels may not accurately represent the true image content. Try your best to determine the topic of the image before proceeding with writing your description.
//                         Step 2: Carefully analyze the title of the image. Extract as much information as possible about the place or possible location, and use your broad knowledge to come up with a nice summary. Don't be shy to use your own research to find more details about the place and location.
//                         Step 3: Write a description that can be used in an Instagram post for the users account. The caption describes the image and captures the essence of the moment. Also, remember that the image was taken from users camera, so you might need to act as a fist person when is needed about the story behind the image or how you captured the moment. This will help the audience connect with the image and understand its significance.
//                         Step 4: Generate hashtags that are relevant to the description and image content. Consider using hashtags that relate to the image, and use engaging and descriptive language. Also, try to generate as many hashtags as possible related to the location, tourist attractions, or parts of the image shown. Hashtags are very important to engage with the audience.
//                         You must follow the following rules:
//                         1. Summarize the capture in a single sentence and ensure that the description and hashtags do not exceed the 2200-character limit. This limit is hardcoded for Instagram captions.
//                         2. Do not use using phrases such as "high probability score", "group of labels", "the object detected", and "score" to represent prediction results.
//                         3. Do not use using time-related words such as "today", "yesterday", "last year", etc., since we do not know when the image was captured.
//                         4. Do not use using words such as "Description:" or "Hashtags:" that explicitly indicate the start of the description or hashtags.
//                         5. The image description should be descriptive and not contain wording such as "The image is most likely to be a mountain …". Instead, it should be something like "Mountain view on a nice summer day with a reflection in the lake …". Use your own imagination to come up with a nice caption. The three dots '...' in the examples indicate that the text should continue.
//                         6. It is good to include a personal touch in your writing. For example, you could say "This is an image I took..." or "This scenery was captured by me..." or "I had the opportunity to take a photo of this great view that I visited...”
//                         7. Prioritize the title of the image over the image content detection since it can be more accurate.
//                         `,
//                     },
//                     {
//                         role: "user",
//                         content: `
//                         ${generateImageClassificationsContent(predictionsClassification)}
//                         ${getObjectDetectionContent(predictionsObjectDetection)}
//                         ${generateTitleContent(fileName)}
//                         `,
//                     },
//                 ],
//             }),
//         });
//         const resultOpenAi = await responseOpenAi.json();
//         document.getElementById("caption")!.innerHTML = resultOpenAi?.choices?.[0]?.message?.content ?? "";
//     }
// }
