

<!-- Mathjax -->
<script type='text/x-mathjax-config'>
		MathJax.Hub.Config({
			extensions: ['tex2jax.js'],
			jax: ['input/TeX', 'output/HTML-CSS'],
			tex2jax: {
				inlineMath: [ ['$','$'], ['\\(','\\)'] ],
				displayMath: [ ['$$','$$'], ['\\[','\\]'] ],
				processEscapes: true
			},
			'HTML-CSS': { availableFonts: ['TeX'] }
		});
</script>

<!-- The following code is for the quizzes -->
<script src="https://surveyjs.azureedge.net/1.0.50/survey.jquery.js"></script>
<link href="https://surveyjs.azureedge.net/1.0.50/survey.css" type="text/css" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/showdown/1.6.4/showdown.min.js"></script>  

<!-- Various toggle functions used throughout --> 
<script language="javascript">
function toggle(id1,id2) {
	var ele = document.getElementById(id1); var text = document.getElementById(id2);
	if (ele.style.display == "block") {ele.style.display = "none"; text.innerHTML = "Show Solution";}
		else {ele.style.display = "block"; text.innerHTML = "Hide Solution";}}
function togglecode(id1,id2) {
   var ele = document.getElementById(id1); var text = document.getElementById(id2);
   if (ele.style.display == "block") {ele.style.display = "none"; text.innerHTML = "Show R Code";}
      else {ele.style.display = "block"; text.innerHTML = "Hide R Code";}}
function toggleEX(id1,id2) {
   var ele = document.getElementById(id1); var text = document.getElementById(id2);
   if (ele.style.display == "block") {ele.style.display = "none"; text.innerHTML = "Show Example";}
      else {ele.style.display = "block"; text.innerHTML = "Hide Example";}}
function toggleTheory(id1,id2) {
   var ele = document.getElementById(id1); var text = document.getElementById(id2);
   if (ele.style.display == "block") {ele.style.display = "none"; text.innerHTML = "Show Theory";}
      else {ele.style.display = "block"; text.innerHTML = "Hide Theory";}}
function toggleQuiz(id1,id2) {
   var ele = document.getElementById(id1); var text = document.getElementById(id2);
   if (ele.style.display == "block") {ele.style.display = "none"; text.innerHTML = "Show Quiz Solution";}
      else {ele.style.display = "block"; text.innerHTML = "Hide Quiz Solution";}}      
function toggleText(id1,id2) {
   var ele = document.getElementById(id1); var text = document.getElementById(id2);
   if (ele.style.display == "block") {ele.style.display = "none"; text.innerHTML = "Show Text";}
      else {ele.style.display = "block"; text.innerHTML = "Hide Text";}}   
</script>

<!-- A few functions for revealing definitions -->
<script language="javascript">
<!--   $( function() {
    $("#tabs").tabs();
  } ); -->

$(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip();
});

$(document).ready(function(){
    $('[data-toggle="popover"]').popover(); 
});
</script>

<script language="javascript">
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Get the element with id="defaultOpen" and click on it
document.getElementById("defaultOpen").click();
</script>


