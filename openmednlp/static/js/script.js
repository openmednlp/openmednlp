$(function () {
    console.log('ready');
    $('#process').on('click', function(e) {
        console.log('process clicked')
        input_text = $('#input_text').serialize();
        console.log(input_text);
        $.ajax({
            type: 'POST',
            url: 'process',
            data: input_text,
        }).done(function(data) {
            console.log('got')
            console.log(data.result);
            $('#result').text(data.input);
            $('#result_status').text(data.status);
        }).fail(function(error) {
            console.error(error);
        })
        $('#result').html()
    });
});