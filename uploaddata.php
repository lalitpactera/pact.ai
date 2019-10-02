<!Doctype html> 
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <title>Solution </title>
        <link rel="stylesheet" href="css/normalize.css">
        <link rel="stylesheet" href="css/style.css">
    </head>
    <body>
        <!-- Logo and home button -->
        <div class="pact">
			<div class='box-scene '>
				
					<div class='box'>
						<div class='front face voice_intelli'>
							<img src='img/Asset 12logo.png' alt='Pactai logo' class="pact-logo ">
						</div>
						<div class="side face">
							<a href = "index.php"><img src='img/home.png' alt='vision' class=" home" ></a>
						</div>
					</div>
				
			</div>  
		</div>

     <div class="heading-text"> 
            <h1>Upload Data</h1> <!--Chest X-ray Image Classification.</h1>-->
        </div>
     <div id="wrapper">
          <form action="uploaddata_backend.php" method="post" id="form" enctype="multipart/form-data">
            <ul class="">
                
                      <li>
                            <h3>Upload Data</h3>
                            <p>Enter only one file at time</p>
                      </li>
                      <li>
                          <input type="file" name="fileToUpload" id="fileToUpload" required >
                      </li>
                        <li>
                            <button name="submit" class="submit">
                               upload
                            </button>
                        </li>

            </ul>

          </form>
      
     </div>

    </body>
</html>