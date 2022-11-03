document.getElementById('smit').onclick = function(){ 	

var budget = document.querySelector("[name='budget']").value;
var ppm = document.querySelector("[name='ppm']").value;
var node_id = document.querySelector("[name='pubkey']").value;
var minimum_capacity = document.querySelector("[name='min-cap']").value;
var maximum_capacity = document.querySelector("[name='max-cap']").value;
var minimum_channels = document.querySelector("[name='min-chan-count']").value;
var min_avg_capacity = document.querySelector("[name='min-avg-cap']").value;

var payload = {
    "env":{
        "budget": budget,
        "ppm": ppm,
        "node_id": node_id
    },
    "agent":{
        "type":"a2c"
    },
    "edge_filters":{
        "minimum_capacity": minimum_capacity,
        "maximum_capacity": maximum_capacity
    },
    "action_mask":{
        "minimum_channels": minimum_channels,
        "min_avg_capacity": min_avg_capacity
    }
}


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

    document.getElementById('result').innerHTML = JSON.stringify(JSON.stringify(data));
console.log({data})
  }).catch(function (err) {
	// There was an error
	console.warn('Something went wrong.', err);
});




}

