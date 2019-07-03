<!DOCTYPE html>
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
			<a href="index.php">
				<div class='box'>
					<div class='front face voice_intelli'>
						<img src='img/Asset 12logo.png' alt='Pactai logo' class="pact-logo ">
					</div>
					<div class="side face">
						<img src='img/home.png' alt='vision' class=" home" >
					</div>
				</div>
			</a>
		</div>  
     </div>
     <div class="heading-text"> 
            <h1>Results</h1>
        </div>

        <div class="results">
            <ul>
                <li>
                        <div class="image">
                                <img src="<?php echo $_GET['p6']; ?>" alt="ai iamge">
                            </div>
                </li>
                <li>
                        <div class="tables">
                                <table>
                                    <tr>
                                        <td><?php echo $_GET['p1']; ?>
                                            </td>
                                        <td><?php echo $_GET['p2']; ?>
                                            </td>
                                    </tr>
                                    <tr>
                                            <td><?php echo $_GET['p3']; ?>
                                                </td>
                                            <td><?php echo $_GET['p4']; ?>
                                                </td>
                                        </tr>
                                </table>
                            </div>
                </li>
                <li>
                    <div class="printing">
                            <button name="print" class="print head">
                                    Print
                                 </button><a href=<?php echo $_GET['p5']; ?>><button name="download" class="download head">
                                         download
                                      </button></a>
                    </div>
                       
                    </li>
                    
            </ul>
            
            
        </div>
    </body>
</html>