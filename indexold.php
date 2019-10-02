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
            <h1>Data pipeline</h1>
        </div>
     <div id="wrapper">
          <form action="backend_workbench.php" method="post" id="productDetailForm2" enctype="multipart/form-data">
            <ul class="">

                      <li>
                          <input type="file" name="fileToUpload" id="fileToUpload">Read data
                      </li>
                      <li>
                          <input type="checkbox" name="task[]" value="A">Data imputation
                      </li>
					  <li>
                          <input type="checkbox" name="task[]" value="B">Data augmentation
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="C">Remove duplications
                      </li>
                      <li>
                          <input type="checkbox" name="task[]" value="D">Remove nans
                      </li>
					  <li>
                          <input type="checkbox" name="task[]" value="E">Normalization
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="F">Categorical to numeric
                      </li>
					  <li>
                          <input type="checkbox" name="task[]" value="G">Stemming and cleaning
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="H">Logistic regression
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="I">Linear regression
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="J">Random forest
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="K">XG Boast
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="L">Support vector machine classifier
                      </li>
                      <li>
                            <input type="checkbox" name="task[]" value="M">Support vector machine regression
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="N">Decision tree classifier
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="O">Decision tree regression
                      </li>
					  <li>
                            <input type="checkbox" name="task[]" value="P">ROC curve
                      </li>
					  <li>
                            <button name="submit" class="submit">
                               upload and submit
                            </button>
                        </li>
            </ul>

          </form>
      
     </div>

    </body>
</html>