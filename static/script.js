document.getElementById('smit').onclick = function() {

    var loaderElm = document.getElementById("loader");
    var resultElm = document.getElementById("result");

    // Error message fields
    var pubkeyMsg = document.getElementById("pubkey-msg");
    var ppmMsg = document.getElementById("ppm-msg");
    var budgetMsg = document.getElementById("budget-msg");
    var minCapMsg = document.getElementById("min-cap-msg");
    var maxCapMsg = document.getElementById("max-cap-msg");
    var minChanMsg = document.getElementById("min-chan-count-msg");
    var minAvgCapMsg = document.getElementById("min-avg-cap-msg");

    // Field Values
    var budget = document.querySelector("[name='budget']").value;
    var ppm = document.querySelector("[name='ppm']").value;
    var node_id = document.querySelector("[name='pubkey']").value;
    var minimum_capacity = document.querySelector("[name='min-cap']").value;
    var maximum_capacity = document.querySelector("[name='max-cap']").value;
    var minimum_channels = document.querySelector("[name='min-chan-count']").value;
    var min_avg_capacity = document.querySelector("[name='min-avg-cap']").value;

    function validateForm() {
        var isValid = true;
        if (budget > 100) {
            budgetMsg.innerText = "Enter a number less than 100";
            isValid = false;
        }
        if (budget < 0) {
            budgetMsg.innerText = "Budget cannot be less then zero";
            isValid = false;
        }
        if (minimum_capacity > maximum_capacity) {
            minCapMsg.innerText = "Min capacity must be lower than max capacity";
            maxCapMsg.innerText = "Max capacity must be higher than min capacity";
            isValid = false;
        }
        if (ppm > 100000) {
            ppmMsg.innerText = "Fee PPM must be less than 100,000";
            isValid = false;
        }
        if (ppm < 0) {
            ppmMsg.innerText = "Fee PPM cannot be less than zero";
            isValid = false;
        }
        if (minimum_capacity < 0) {
            minCapMsg.innerText = "Minimum capacity cannot be less than zero";
            isValid = false;
        }
        if (minimum_capacity > 1000000000) {
            minCapMsg.innerText = "Minimum capacity cannot be greater than 10 BTC (1 billion sats)";
            isValid = false;
        }
        if (maximum_capacity < 0) {
            maxCapMsg.innerText = "Maximum capacity cannot be less than zero";
            isValid = false;
        }
        if (maximum_capacity > 10000000000) {
            maxCapMsg.innerText = "Maximum capacity cannot be greater than 10 BTC (1 billion sats)";
            isValid = false;
        }
        if (minimum_channels < 1) {
            minChanMsg.innerText = "Minimum channels must be greater than zero"
            isValid = false;
        }
        if (min_avg_capacity < 20000) {
            minAvgCapMsg.innerText = "Must be greater then 20,000";
            isValid = false;
        }
        return isValid;
    }

    var payload = {
        "env": {
            "budget": budget,
            "ppm": ppm,
            "node_id": node_id
        },
        "agent": {
            "type": "a2c"
        },
        "edge_filters": {
            "minimum_capacity": minimum_capacity,
            "maximum_capacity": maximum_capacity
        },
        "action_mask": {
            "minimum_channels": minimum_channels,
            "min_avg_capacity": min_avg_capacity
        }
    }

    function getResults(data) {
        var resultHtml = `<h3>Opening these channels will give you a betweenness score of:</h3><h3>${data.betweenness}</h3> <h4>Higher is better.</h4><ol>`;
        for (var rec of Object.keys(data.recommendations)) {
            resultHtml += `<li><a href="https://amboss.space/node/${data.recommendations[rec]}">${rec || data.recommendations[rec]}</a></li>`
        }

        return resultHtml + "</ol>";

    }

    if (validateForm()) {

        loaderElm.className = "";
        resultElm.className = "hidden";

        fetch('/api', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        }).then(response => {

            if (response.ok) {
                console.log("OK")
                return response.json();
            } else {
                return Promise.reject(response);
            }


        }).then(data => {
            loaderElm.className = "hidden";
            resultElm.className = "";
            resultElm.innerHTML = getResults(data);
            console.log({
                data
            })
        }).catch(function(err) {
            // There was an error
            loaderElm.className = "hidden";
            console.warn('Something went wrong.', err);
        });

    } // end if validate

}
