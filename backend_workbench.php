<?php 
ini_set('max_execution_time', 3000);

$data_task = $_POST['task'];
if(empty($data_task))
{    
	echo("You didn't select any option.");
}
else
{
	echo implode($data_task);
}

$newFilePath = $_FILES["fileToUpload"]["name"];

if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $newFilePath)) {
    #echo "The file ". basename( $_FILES["fileToUpload"]["name"]). " has been uploaded.";
	;
} else {
    #echo "Sorry, there was an error uploading your file.";
	;
}

$str1 = 'python backend_workbench.py ' . $newFilePath . ' ' . implode($data_task);

$str2 = shell_exec($str1);

$str1 = 'Location: result_workbench.php';
#echo($str1);
header($str1);
exit();
?>