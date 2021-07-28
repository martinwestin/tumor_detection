String.prototype.format = function () {
    var i = 0, args = arguments;
    return this.replace(/{}/g, function () {
        return typeof args[i] != 'undefined' ? args[i++] : '';
    });
};

let socket;
$(document).ready(function() {
  socket = io.connect("/");

  $("#submit-button").click(function() {
    const selected = document.getElementById("pic").files[0];
    socket.emit("classify_image", {
      data: selected
    });
  });

  socket.on("classification_response", function(msg) {
    const resultDiv = $("#result-text-container");
    let newInnerTextHTML = "<p>Diagnosis: <strong>{}</strong>".format(msg["result"]);
    newInnerTextHTML += "<br><br>Remember that this is not professional health care, and the results given are not 100% accurate.</p>";
    resultDiv.html(newInnerTextHTML);
  });
});
