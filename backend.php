<?php 
ini_set('max_execution_time', 3000);

$user_task  = $_GET['task']; 
$readfromfilevar = "0";
$target = $_POST['target'];
$task_end = 0;

if ($user_task == 5)
	$task_end = $_POST['selector'];
	
if ($user_task > 1 and $user_task < 5)
{
	$data = $_POST['data'];
	$features = 'a';
}
if ($user_task == 5)
{
	$features = $_POST['features'];
}
if ($user_task == 6)
{
	$data = $_POST['data'];
}

if ($target == ''){
	$target = '-';
}

if ($features == ''){
	$features = '-';
}

#echo $task_end;
#echo $features;
#echo $target;

$newFilePath = $_FILES["fileToUpload"]["name"];

if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $newFilePath)) {
    #echo "The file ". basename( $_FILES["fileToUpload"]["name"]). " has been uploaded.";
	;
} else {
    #echo "Sorry, there was an error uploading your file.";
	$readfromfilevar = "1";
	$myfile = fopen("features.txt", "w");
	fwrite($myfile, $data);
	fclose($myfile);
	$newFilePath = 'dummy.csv';
}

$str1 = 'python task_exec.py ' . $user_task . ' ' . $newFilePath . ' ' . $readfromfilevar . ' ' . $features . ' ' . $target . ' ' . $task_end;
#echo($str1);
$str2 = shell_exec($str1);

if ($user_task == 5)
{
	header('Location: index.php');
	exit();
}

$file = fopen("results.txt","r");
$name1 =  rtrim(fgets($file));
$name2 =  rtrim(fgets($file));
$name3 =  rtrim(fgets($file));
$name4 =  rtrim(fgets($file));
$name5 =  rtrim(fgets($file));
$name6 =  fgets($file);
fclose($file);

$str1 = 'Location: result.php?p1='.$name1.'&p2='.$name2.'&p3='.$name3.'&p4='.$name4.'&p5='.$name5.'&p6='.$name6;
#echo($str1);
header($str1);
exit();
?>