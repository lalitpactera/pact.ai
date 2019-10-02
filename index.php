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
        <title>PACT AI</title>
<link rel="stylesheet" href="css/homepage.css">
</head>
<body>
<div class="grid-container">
  <div class="item1">
    <div class="logo">
        <div class="pact">
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
      </div>


  </div>
  <div class="item2">
      <h1>Various offerings from pactera</h1>
  </div>
  <div class="item3">
    <div class="catlog">
        <!-- box -1 -->
        <div class='box-scene'>
          <div class='box'>
              <div class='front face'>
                  <img src="img/Asset 4vision.png" alt="vision logo" class="icon-assets">
                  <p>Chest X-ray image classification</p>
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
              <img src="img/text.png" alt="voice logo" class="icon-assets">
              <p>Tickets classification</p>
              <!-- <img src='http://placehold.it/180x180/' alt=''> -->
          </div>
          <div class="side face">
              <!-- <p>This is back</p> -->
              <a href = "solution_classification_regression.php?title=Tickets parent-id classification&task=2&features=History&target=Parent"><img src='img/text clasify.png' alt='vision'></a>
          </div>
      </div>
  </div>
  <!-- box 3 -->
   <div class='box-scene'>
          <div class='box'>
              <div class='front face brain_intelli'>
                  <img src="img/finance.png" alt="Brain logo" class="brain-img" >
                  <p>Price prediction</p>
                  <!-- <img src='http://placehold.it/180x180/' alt=''> -->
              </div>
              <div class="side face">
                  <!-- <p>This is back</p> -->
                  <a href="solution_classification_regression.php?title=Price prediction&task=3&features=Year,Month,Units,Product_Level1&target=Price"><img src='img/money.png' alt='vision'></a>
              </div>
          </div>
      </div>
      <!-- box 4 -->
<div class='box-scene'>
      <div class='box'>
          <div class='front face'>
              <img src="img/AR.png" alt="robot logo" class="robot">
              <p>Sales prediction</p>
              <!-- <img src='http://placehold.it/180x180/' alt=''> -->
          </div>
          <div class="side face">
              <!-- <p>This is back</p> -->
              <a href="solution_classification_regression.php?title=Sales prediction&task=4&features=Supplies Subgroup,Supplies Group,Region,Route To Market,Elapsed Days In Sales Stage,Sales Stage Change Count,Total Days Identified Through Closing,Total Days Identified Through Qualified,Opportunity Amount USD,Client Size By Revenue, Client Size By Employee Count,Revenue From Client Past Two Years,Competitor Type,Ratio Days Identified To Total Days,Ratio Days Validated To Total Days,Ratio Days Qualified To Total Days,Deal Size Category&target=Opportunity Result"><img src='img/Ar-girl.png' alt='vision'></a>
          </div>
      </div>
  </div>

  <!-- next row of boxes -->
    <!-- box -1 -->
    <div class='box-scene'>
            <div class='box'>
                <div class='front face voice_intelli'>
                    <img src="img/Asset 3mic.png" alt="vision logo" class="icon-assets">
                    <p>PII</p>
                    <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                </div>
                <div class="side face">
                    <!-- <p>This is back</p> -->
                    <a href = "solution_audio_classification.php?title=Audio to text + PII&task=7&features=a&target=a"><img src='img/voice.png' alt='vision'></a>
                </div>
            </div>
        </div>
  <!-- box -2 -->
  <div class='box-scene'>
        <div class='box'>
            <div class='front face brain_intelli'>
                <img src="img/logistics.png" alt="voice logo" class="icon-assets">
                <p>Safety app - heavy industry</p>
                <!-- <img src='http://placehold.it/180x180/' alt=''> -->
            </div>
            <div class="side face">
                <!-- <p>This is back</p> -->
                <a href="solution_classification_regression.php?title=Safety app - heavy industry&task=8&features=Incident Description&target=Record Classification"><img src='img/cargo.png' alt='vision'></a>
            </div>
        </div>
    </div>
    <!-- box 3 
     <div class='box-scene'>
            <div class='box'>
                <div class='front face brain_intelli'>
                    <img src="img/Asset 2robot1.png" alt="Brain logo" class="brain-img" >
                    <p>Knowledge Intelligence</p>
                    
                </div>
                <div class="side face">
                    
                    <img src='img/robot.png' alt='vision'>
                </div>
            </div>
        </div>
        <!-- box 4 
  <div class='box-scene'>
        <div class='box'>
            <div class='front face'>
                <img src="img/Asset 2robot1.png" alt="robot logo" class="robot">
                <p>Robotic process automation</p>
                
            </div>
            <div class="side face">
                
                <img src='img/robot.png' alt='vision'>
            </div>
        </div>
    </div>-->


</div>

  </div>  
  <div class="item4">
        <div class="catlog">
                <!-- box -1 -->
                <div class='box-scene'>
                  <div class='box'>
                      <div class='front face'>
                          <img src="img/brain.png" alt="vision logo" class="icon-assets">
                          <p>Train your model</p>
                          <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                      </div>
                      <div class="side face">
                          <!-- <p>This is back</p> -->
                          <a href="solution.html"><img src='img/brain-intelli.png' alt='Ar girl'></a> 
                      </div>
                  </div>
              </div>
        <!-- box -2 -->
        <div class='box-scene'>
              <div class='box'>
                  <div class='front face voice_intelli'>
                      <img src="img/brain.png" alt="voice logo" class="icon-assets">
                      <p>Test your model</p>
                      <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                  </div>
                  <div class="side face">
                      <!-- <p>This is back</p> -->
					  <a href="solution_classification_regression.php?title=Test your model&task=6&features=<?php echo $name1; ?>&target=<?php echo $name2; ?>"><img src='img/brain-intelli.png' alt='vision'></a>
                  </div>
              </div>
          </div>

    <!-- box -3 -->
    <div class='box-scene'>
              <div class='box'>
                  <div class='front face brain_intelli'>
                      <img src="img/uploadicon.png" alt="voice logo" class="icon-assets">
                      <p>Upload Data</p>
                      <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                  </div>
                  <div class="side face">
                      <!-- <p>This is back</p> -->
					  <a href="uploaddata.php"><img src='img/uploadimg.png' alt='vision'></a>
                  </div>
              </div>
          </div>

    <!-- box -4 -->
    <div class='box-scene'>
              <div class='box'>
                  <div class='front face'>
                      <img src="img/aiicon.png" alt="voice logo" class="icon-assets">
                      <p>Data pipeline and modeling</p>
                      <!-- <img src='http://placehold.it/180x180/' alt=''> -->
                  </div>
                  <div class="side face">
                      <!-- <p>This is back</p> -->
					  <a href="sandbox.php"><img src='img/aiimg.png' alt='vision'></a>
                  </div>
              </div>
          </div>

          <!-- box 3 
           <div class='box-scene'>
                  <div class='box'>
                      <div class='front face brain_intelli'>
                          <img src="img/Asset 2robot1.png" alt="Brain logo" class="brain-img" >
                          <p>Knowledge Intelligence</p>
                      </div>
                      <div class="side face">
                          <img src='img/robot.png' alt='vision'>
                      </div>
                  </div>
              </div>
              <!-- box 4 
        <div class='box-scene'>
              <div class='box'>
                  <div class='front face'>
                      <img src="img/Asset 2robot1.png" alt="robot logo" class="robot">
                      <p>Robotic process automation</p>
                   
                  </div>
                  <div class="side face">
                 
                      <img src='img/robot.png' alt='vision'>
                  </div>
              </div>
          </div>-->
        </div>
        
  </div>
  <div class="item5">All rights are reserved by pactera</div>
</div>

</body>
</html>

