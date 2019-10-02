<?php
$json = file_get_contents('parameters.json');
$json_data = json_decode($json,true);
$features = implode(',', $json_data['features']);
$target = $json_data['target'][0];
$lr_c = $json_data['logisticregression_c'];
$lr_auto = $json_data['logisticregression_auto'];
$norm_mmstd = $json_data['normalization'];
if ($norm_mmstd == 'minmax')
{
	$norm_mm = 'selected';
	$norm_std = '';
}
else
{
	$norm_mm = '';
	$norm_std = 'selected';
}
$split = $json_data['logisticregression_split'];
?>

<!-- <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en"> -->
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8" />
    <meta http-equiv="content-script-type" content="text/javascript" />
	<meta http-equiv="content-style-type" content="text/css" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta http-equiv="X-UA-Compatible" content="ie=edge">

    <title>Pact AI drag & Drop</title>
    <meta name="description" content="Where have I been?" />

    <script src="http://www.google.com/jsapi" type="text/javascript"></script>
	<script type="text/javascript">
	    google.load("jquery", "1.7.1");
		google.load("jqueryui", "1.7.2");
	</script>
	<link rel="stylesheet" type="text/css" href="css/style_wb.css" media="all" />
</head>

<body>
	<div class="grid-container">
	<div class="item1">
		<div class="logo">
			<div class="pact">
					<div class='box-scene '>
					   <div class='box'>
						 <div class='front face voice_intelli'>
						 <img src='images/Asset 12logo.png' alt='Pactai logo' class="pact-logo ">
					  </div>
				<div class="side face">
						<a href="index.php"><img src='images/home.png' alt='vision' class=" home" ></a>
				</div>
			</div>
		</div>  
		 </div>
		  </div>
	
	
	  </div>

<div id="wrapper">
	<div id="options">
		<div id="drag12" class="drag"><p>Start</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>In preperation</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag12 -->

		<div id="drag1" class="drag "><p>Select Data</p><div class='modal'><div class="modal-content"><div class="modal-header">
			<span class="close">&times;</span>
			<h2>Options</h2>
		  </div>
		  <div class="modal-body">
			<select name ="select_form" id="select_form" size="5" onchange="update_select(this)">
				<?php 
					foreach (glob("dataset\*.csv") as $filename) { 
					echo("<option value=\""  . basename($filename) .  "\">" . basename($filename) . "</option> <br>");
					}
					foreach (glob("dataset\*.txt") as $filename) { 
						echo("<option value=\""  . basename($filename) .  "\">" . basename($filename) . "</option> <br>");
					}
				?>
			</select>
			<p>
			<button type="button" class="lr_save" id = "select_save">Ok</button>
			</p>
		  </div>
		  <div class="modal-footer">
			<h3></h3>
		  </div></div><div class="close">X</div></div>
		  </div> <!-- end of drag1 -->
		
		<div id="drag2" class="drag "><p>Remove Duplications</p> <div class='modal'><div class="modal-content"><div class="modal-header">
			<span class="close">&times;</span>
			<h2>Options</h2>
		  </div>
		  <div class="modal-body">
			<p></p>
		  </div>
		  <div class="modal-footer">
			<h3></h3>
		  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag2 -->

		<div id="drag3" class="drag"><p>Remove Missing Data</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p></p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag3 -->

		<div id="drag4" class="drag"><p>SQL</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>Enter Query (Enter table name as df, eg. SELECT * FROM df)</p>
				<textarea name="query" id="query" cols="60" rows="5" onkeyup="update_query(this)"></textarea>
				<p>
				<button type="button" class="lr_save" id = "query_save">Ok</button>
				</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag4 -->
		
		<div id="drag5" class="drag"><p>Normalization</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<h1></h1>
				<select name ="normalization_form" id="normalization_form" size="2" onchange="update_norm(this)">
					<option value="minmax" <?php echo $norm_mm; ?>>MinMax</option>
					<option value="stdnorm" <?php echo $norm_std; ?>>Standard</option>
				</select>
				<h1></h1>
				<p>
				<button type="button" class="lr_save" id="normalization_save">Ok</button>
				</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag5 -->
		
		<div id="drag6" class="drag"><p>Categorical to Numeric</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p></p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag6 -->
		
		<div id="drag7" class="drag"><p>Join</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>
				</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag7 -->
		
		<div id="drag8" class="drag"><p>Data Imputation</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>Currently using mean (only)</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag8 -->
		
		<div id="drag9" class="drag"><p>Logistic Regression</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				  <div class="nodal-textbox">
                    <p>Column names (features)</p>
					<textarea name="features" id="features" cols="60" rows="5" onkeyup="update_features(this)"><?php $json = file_get_contents('parameters.json'); $json_data = json_decode($json,true); $features = implode(',', $json_data['features']); echo $features; ?></textarea>
					<p>Column name (target)</p>
					<textarea name="target" id="target" cols="60" rows="2" onkeyup="update_target(this)"><?php $json = file_get_contents('parameters.json'); $json_data = json_decode($json,true); $target = $json_data['target'][0]; echo $target; ?></textarea>
					<p>C value</p>
					<input type="text" name="cvalue" id = "cvalue" onkeyup="update_cvalue(this)" value=<?php echo $lr_c; ?>></input>
					<p>Split%</p>
					<p>
					<input type="text" name="cvalue" id = "split" pattern="[0-9]" onkeyup="update_split(this)" value=<?php echo $split; ?>></input>
					</p>
					<p>
					<input type="checkbox" name="auto" id = "auto" onClick="update_auto(this)" value = 'T' <?php echo $lr_auto; ?>>Auto<br>
					</p>
					<p>
					<button type="button" class="lr_save" id = "lr_save">Ok</button>
					</p>
					</div>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag9 -->
		
		<div id="drag10" class="drag"><p>Test Model</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<select name ="select_form" id="test_form" size="5" onchange="update_test(this)">
				<?php 
					foreach (glob("*.csv") as $filename) { 
					echo("<option value=\""  . $filename .  "\">" . $filename . "</option> <br>");
					}
				?>
			</select>
			<p>
			<button type="button" class="lr_save" id = "test_save">Ok</button>
			</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag10 -->
		
		<div id="drag11" class="drag"><p>Save</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>Use this to branchof result</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag11 -->
		

		
		<div id="drag13" class="drag"><p>Grammer Correction</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>In preperation</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag13 -->
		
		<div id="drag14" class="drag"><p>NER</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>In preperation</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag14 -->
		
		<div id="drag15" class="drag"><p>MaskPII</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
			  <p><input type="checkbox" name="serialnumbers" id = "serialnumbers" onClick="update_serialnumbers(this)">Serial Numbers<br></p>
			  <p><input type="checkbox" name="numbers" id = "numbers" onClick="update_numbers(this)">Numbers (5+ digits)<br></p>
			  <p><input type="checkbox" name="fullnames" id = "fullnames" onClick="update_fullnames(this)">Full Names<br></p>
			  <p><input type="checkbox" name="organizations" id = "organizations" onClick="update_organizations(this)">Organizations<br></p>
			  <p><input type="checkbox" name="emailids" id = "emailids" onClick="update_emailids(this)">Email ids<br></p>
			  <p><input type="checkbox" name="urls" id = "urls" onClick="update_urls(this)">URLs<br></p>
			  <p><input type="checkbox" name="fulldates" id = "fulldates" onClick="update_dates(this)">Dates<br></p>
			  <p><button type="button" class="lr_save" id = "maskpii_save">Ok</button></p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div> <!-- end of drag15 -->
		
		<div id="drag16" class="drag"><p>Stemming and cleaning</p><div class='modal'><div class="modal-content"><div class="modal-header">
				<span class="close">&times;</span>
				<h2>Options</h2>
			  </div>
			  <div class="modal-body">
				<p>In preperation</p>
			  </div>
			  <div class="modal-footer">
				<h3></h3>
			  </div></div><div class="close">X</div></div>
		</div>
	</div><!-- end of options -->
	<!-- <span id="title"><h2>Drop items here</h2></span> -->
	<div id="frame"></div><!-- end of frame -->
</div><!-- end of wrapper -->
<div class="results">
	<button type="button" class="btnsimulate">Run</button>
	<a href="output\results_0.csv"><button type="button" class="btn">Download</button></a>
	<button type="button" class="btn">Load</button>
	<button type="button" class="btn">Save</button>
</div>
  </div>  <!-- Grid container close--> 
</body>
<script type="text/javascript" src="js/jquery.connectingLine.js"></script>
<script type="text/javascript" src="js/custom.js"></script>
</html>