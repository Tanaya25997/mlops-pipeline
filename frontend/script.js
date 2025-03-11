//waits for the user to click the Upload Image button and then calls uploadImage()

document.getElementById('upload_image').addEventListener('click', uploadImage);

function uploadImage() {

    let input_file = document.getElementById('imageUpload');
    let file_ = input_file.files[0];

    if (!file_) {
        alert("Please upload an Image!");
        return;
    }


    console.log("Selected file : ", file_)

    // prepare image for upload 
    let formData = new FormData()
    formData.append("file", file_)  // wrap the data in a format that the backend can undertsand - in our case file

    // Send it to backend 
    fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())  // Parse the response as JSON
    .then(data => {
        console.log("Response from server:", data);
        // Once you get the response, display the predicted class and confidence
        document.getElementById("result").innerText = "Predicted class: " + data.prediction + ", Confidence: " + data.Confidence + "%";
    })
    .catch(error => {
        console.error("Error:", error);
        // Handle any errors that occur during the request
        document.getElementById("result").innerText = "Error uploading image. Please try again.";
    });


    document.getElementById("feedback_text").innerText = 'Help us imporve our model. Was this prediction correct?';
    document.getElementById("correct-prediction").innerText = 'Yes, the prediction is correct';
    document.getElementById("wrong-prediction").innerText = 'No, the prediction is wrong';



}
