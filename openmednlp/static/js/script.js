$(function () {
    console.log('ready');
    $('#process').on('click', function(e) {
        var input_text = $('#input_text').serialize();
        $.ajax({
            type: 'POST',
            url: 'process',
            data: input_text,
        }).done(function(data) {
            $('#result').text(data.input);
            $('#result_status').text(data.status);
        }).fail(function(error) {
            console.error(error);
        })
        $('#result').html()
    });
});