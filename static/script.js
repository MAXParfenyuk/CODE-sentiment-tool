function predictSentiment() {
    var review = document.getElementById('userInput').value;
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            user_review: review
        })
    })
    .then(response => response.json())
    .then(data => {
        var result = document.getElementById('result');
        result.innerText = `Predicted Sentiment: ${data.prediction}`;
    });
}
