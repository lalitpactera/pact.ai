<?php 
ini_set('max_execution_time', 3000);

$newFilePath = "dataset/" . $_FILES["fileToUpload"]["name"];

if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $newFilePath)) {
    #echo "The file ". basename( $_FILES["fileToUpload"]["name"]). " has been uploaded.";
	;
} 

$str1 = 'Location: index.php';
header($str1);
exit();
?>