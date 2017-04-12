function updateFeatureValues() {
  var featureInputs = $('.feature-input');
  featureInputs.bind('input', function() {
    var value = $(this).attr('value');
    $(this).next().html(value);
  });
}

$(document).ready(updateFeatureValues);
