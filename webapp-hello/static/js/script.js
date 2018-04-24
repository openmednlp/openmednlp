$(function () {

    /*$("#submitButton").click(function () {
        var url = '/hello';
        var inputFormContent = $("#usernameInputField").val();
        var postRes = $.ajax({
            url: url,
            type: 'POST',
            data: JSON.stringify({username: inputFormContent}),
            contentType: 'application/json; charset=utf-8'
        });

    });*/

    $("#submitForm").submit(function(event){
        //event.preventDefault(); method doesn't do anything now. event performed using default --> form

        var url = '/hello';
        var inputFormContent = $("#usernameInputField").val();
        //$.post(url, JSON.stringify({username: inputFormContent}), 'json');
        $.post(url, $("#submitForm").serialize());
    });





});
