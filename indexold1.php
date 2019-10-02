<?php
$file = fopen("MyFile2.txt","r");
$name1 =  rtrim(fgets($file));
$name2 =  rtrim(fgets($file));
fclose($file); 
?>

<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <title>Pact AI </title>
        <link rel="stylesheet" href="css/normalize.css">
        <link rel="stylesheet" href="css/style.css">
    </head>
    <body>
     <div class="banner">
         
 
         <div class="pact">
                <!-- <div class=' pact-logo-1'>
                        <img src="img/Asset 12logo.png" alt="pactai logo" class="pact-logo ">
                        
                    </div> -->
                    <div class='box-scene '>
            <div class='box'>
                <div class='front face voice_intelli'>
                    <img src='img/Asset 12logo.png' alt='Pactai logo' class="pact-logo ">
                </div>
                <div class="side face">
                        <img src='img/home.png' alt='vision' class=" home" >
                </div>
            </div>
        </div>  
         </div>
         <div class="heading-text"> 
                <h1>Pact.ai various offerings</h1>
            </div>
     </div>

     <div class="container">
        <div class = "catlog">
          <!-- box -1 -->
		  
          <div class='box-scene'>
            <div class='box'>
                <div class='front face'>
                    <img src="img/Asset 4vision.png" alt="vision logo" class="icon-assets">
					<p>Chest X-ray image classification</p>
                    <!--<p>Vision Intelligence</p>
                    <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                </div>
                <div class="side face">
                    <!-- <p>This is back</p> -->
                    <a href = "solution_image_classification.php?title=Chest X-ray Image Classification&task=1"><img src='img/vision.png' alt='vision'></a>
                </div>
            </div>
        </div>
		
<!-- box -2 -->
 <div class='box-scene'>
        <div class='box'>
            <div class='front face voice_intelli'>
                <img src="img/Asset 3mic.png" alt="voice logo" class="icon-assets">
				<p>Tickets classification</p>
                <!-- <p>Voice Intelligence</p>
                <!-- <img src='http://placehold.it/180x180/' alt=''> -->
            </div>
            <div class="side face">
                <!-- <p>This is back</p> -->
                <a href = "solution_classification_regression.php?title=Tickets parent-id classification&task=2&features=History&target=Parent"><img src='img/voice.png' alt='vision'></a>
            </div>
        </div>
    </div>
	

    <!-- box 3 -->
    
		<div class='box-scene'>
            <div class='box'>
                <div class='front face brain_intelli'>
                    <img src="img/brain.png" alt="Brain logo" class="brain-img" >
					<p>Price prediction</p>
                    <!-- <p>Knowledge Intelligence</p>
                    <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                </div>
                <div class="side face">
                    <!-- <p>This is back</p> -->
                    <a href="solution_classification_regression.php?title=Price prediction&task=3&features=Year,Month,Units,Product_Level1&target=Price"><img src='img/brain-intelli.png' alt='vision'></a>
                </div>
            </div>
        </div>
	
	
        <!-- box 4 -->
 <div class='box-scene'>
        <div class='box'>
            <div class='front face'>
                <img src="img/Asset 2robot1.png" alt="robot logo" class="robot">
                <p>Sales prediction</p>
				<!-- <p>Robotic process automation</p>
                <!-- <img src='http://placehold.it/180x180/' alt=''> -->
            </div>
            <div class="side face">
                <!-- <p>This is back</p> -->
                <a href="solution_classification_regression.php?title=Sales prediction&task=4&features=Supplies Subgroup,Supplies Group,Region,Route To Market,Elapsed Days In Sales Stage,Sales Stage Change Count,Total Days Identified Through Closing,Total Days Identified Through Qualified,Opportunity Amount USD,Client Size By Revenue, Client Size By Employee Count,Revenue From Client Past Two Years,Competitor Type,Ratio Days Identified To Total Days,Ratio Days Validated To Total Days,Ratio Days Qualified To Total Days,Deal Size Category&target=Opportunity Result"><img src='img/robot.png' alt='vision'></a>
            </div>
        </div>
    </div>


<!-- box -5 -->
    <div class='box-scene'>
        <div class='box'>
            <div class='front face voice_intelli'>
                <img src="img/Asset 3mic.png" alt="voice logo" class="icon-assets">
				<p>PII </p>
                <!-- <p>Voice Intelligence</p>
                <!-- <img src='http://placehold.it/180x180/' alt=''> -->
            </div>
            <div class="side face">
                <!-- <p>This is back</p> -->
                <a href = "solution_image_classification.php?title=Audio to text + PII&task=7&features=a&target=a"><img src='img/voice.png' alt='vision'></a>
            </div>
        </div>
    </div>

<!-- box -6 -->
		<div class='box-scene'>
            <div class='box'>
                <div class='front face brain_intelli'>
                    <img src="img/brain.png" alt="Brain logo" class="brain-img" >
					<p>Safety app - logistics, oil and gas</p>
                    <!-- <p>Knowledge Intelligence</p>
                    <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                </div>
                <div class="side face">
                    <!-- <p>This is back</p> -->
                    <a href="solution_classification_regression.php?title=Safety app - logistics, oil and gas&task=8&features=Incident Description&target=Record Classification"><img src='img/brain-intelli.png' alt='vision'></a>
                </div>
            </div>
        </div>

</div>

<!-- box 7 -->

    <div class='box-scene'>
            <div class='box'>
                <div class='front face voice_intelli    '>
                    <img src="img/AR.png" alt="Augmented reality" class="ar">
					<p>Train your model</p>
                    <!-- <p>Platform augmented intelligence</p>
                    <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                </div>
                <div class="side face">
                    <!-- <p>This is back</p> -->
                   <a href="solution.html"><img src='img/Ar-girl.png' alt='Ar girl'></a> 
                </div>
            </div>
        </div>

<!-- box 8 -->

     <div class='box-scene'>
            <div class='box'>
                <div class='front face brain_intelli'>
                    <img src="img/AR.png" alt="Augmented reality" class="ar" >
                    <p>Test your model</p>
					<!-- <p>Knowledge Intelligence</p>
                    <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                </div>
                <div class="side face">
                    <!-- <p>This is back</p> -->
                    <a href="solution_classification_regression.php?title=Test your model&task=6&features=<?php echo $name1; ?>&target=<?php echo $name2; ?>"><img src='img/Ar-girl.png' alt='vision'></a>
                </div>
            </div>
        </div>
     </div>

<footer>
    <p>powered by <img src="img/Pactera-EDGE.png" alt="pactera logo" class="pactera"></p>
</footer>
    </body>
</html>