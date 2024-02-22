/***
 * * Author: Hank Rugg
 * Date: Feb. 15, 2024
 *
 * This is the class that will allow us to train a dataset using the nearest neighbors algorithm
 *
 * Resources:
 * For code formatting : https://facebook.github.io/ktfmt/
 * For syntax using dataframes : https://kotlin.github.io/dataframe/overview.html#syntax
 * Returning 2 values from fun() using Pair<>: https://www.baeldung.com/kotlin/returning-multiple-values
 * Ifs and Looping : https://kotlinlang.org/docs/control-flow.html#when-expression
 * Read csv : https://www.baeldung.com/kotlin/csv-files
 *
 */

import java.io.File
import java.io.InputStream
import kotlin.random.Random
import kotlin.math.abs

class NearestNeighbors() {

    var data = mutableListOf<Flower>()
    val test = mutableListOf<Flower>()
    val train = mutableListOf<Flower>()

    // flower data class to store the data from csv
    data class Flower(
        val SepalLengthCm: Float,
        val SepalWidthCm: Float,
        val PetalLengthCm: Float,
        val PetalWidthCm: Float,
        val Species: String,
    )

    // non used method because we arent evaluating our model yet, just making predictions
    fun test_train_split(pctTest :Float): Boolean {
        if (pctTest < 1F) {
            return false
        }
        if (pctTest > 0F){
            return false
        }
        for (flower : Flower in data) {
            val x = Random.nextInt(100)/100F
            if (x < pctTest) {
                test += flower
            } else {
                train += flower
            }
        }
        return true

    }

    fun validateFloatInput(input: String): Boolean {
        return input.toFloatOrNull() != null
    }

    // chat gpt helped with this function
    fun findMode(list: List<String>): String {
        val occurrences = mutableMapOf<String, Int>()
        // Count occurrences of each element in the list
        for (element in list) {
            val count = occurrences.getOrDefault(element, 0)
            occurrences[element] = count + 1
        }
        // Find the maximum occurrence count
        val maxCount = occurrences.values.maxOrNull() ?: 0
        // Find elements with the maximum occurrence count
        val modeElements = occurrences.filterValues { it == maxCount }.keys.toList()
        // Convert list of mode elements to a single string
        return modeElements.joinToString(", ")
    }

    fun loadData(filePath: String){
        //  load data and set it to the data variable
        val file = File(filePath)
        val inputStream: InputStream = file.inputStream()
        data = readCsv(inputStream).toMutableList()
    }

    fun readCsv(inputStream: InputStream): List<Flower> {
        /// read the data into a list
        val reader = inputStream.bufferedReader()
        val header = reader.readLine()
        return reader.lineSequence()
            .filter { it.isNotBlank() }
            .map {
                val (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species) = it.split(",", ignoreCase = true, limit = 5)
                Flower(SepalLengthCm.toFloat(), SepalWidthCm.toFloat(), PetalLengthCm.toFloat(), PetalWidthCm.toFloat(), Species)
            }.toList()
    }

    fun test(numNeighbors : Int, SepalLengthCm: Float, SepalWidthCm: Float, PetalLengthCm: Float, PetalWidthCm: Float): String {
        // initialize list of flowers
        val neighbors = mutableListOf<Flower>()
        // for each flower, see how far away each variable is and make a flower object with the difference
        for (flower: Flower in data) {
            neighbors += Flower(
                abs(flower.SepalLengthCm - SepalLengthCm),
                abs(flower.SepalWidthCm - SepalWidthCm),
                abs(flower.PetalLengthCm - PetalLengthCm),
                abs(flower.PetalWidthCm - PetalLengthCm),
                flower.Species
            )
        }
        // initialize a results map
        var results =  mutableMapOf<Float , String>();
        for (neighbor: Flower in neighbors) {
            // add the absolute difference of each variable as the key and the species as the value
            val sum : Float = neighbor.SepalWidthCm + neighbor.SepalLengthCm + neighbor.PetalLengthCm + neighbor.PetalWidthCm
            results[sum] = neighbor.Species
        }
        // sort the map so the smallest values are first, meaning that the neighbors that are closest are first
        results = results.toSortedMap()

        // get the list of keys which is the distance to each neighbor
        val nums = results.keys.toList()

        // initialize a list for final results
        val finalResults = mutableListOf<String>()

        // for the number of neighbors, for each distance, add the related species to the final results
        for (i in 1..numNeighbors){
            finalResults += results[nums.get(i)].toString()
        }

        println("The neighbors predict:")
        // this will be a list of what the neighbors predict
        println(finalResults)

        // since this is classification, we will find the mode of the predictions from the neighbors and use that as our prediction
        return findMode(finalResults)
    }

}

fun main(args: Array<String>) {
    val n = NearestNeighbors()
    n.loadData("src/main/kotlin/Iris.csv")

    var continueLoop = true

    while (continueLoop) {

        // sepal length
        println("What is the sepal length of your Iris in cm?")
        var slengthInput = readLine()!!
        while (!n.validateFloatInput(slengthInput)) {
            println("Invalid input. Please enter a decimal")
            slengthInput = readLine()!!
        }
        var slength = slengthInput.toFloat()

        // sepal width
        println("What is the sepal width of your Iris in cm?")
        var swidthInput = readLine()!!
        while (!n.validateFloatInput(swidthInput)) {
            println("Invalid input. Please enter a decimal")
            swidthInput = readLine()!!
        }
        var swidth = swidthInput.toFloat()

        // petal length
        println("What is the petal length of your Iris in cm?")
        var plengthInput = readLine()!!
        while (!n.validateFloatInput(plengthInput)) {
            println("Invalid input. Please enter a decimal")
            plengthInput = readLine()!!
        }
        var plength = plengthInput.toFloat()

        // petal width
        println("What is the petal width of your Iris in cm?")
        var pwidthInput = readLine()!!
        while (!n.validateFloatInput(pwidthInput)) {
            println("Invalid input. Please enter a decimal")
            pwidthInput = readLine()!!
        }
        var pwidth = pwidthInput.toFloat()

        // number of neighbors
        println("What is the number of neighbors you would like to train on?")
        var neighborsInput = readLine()!!
        var numNeighbors = neighborsInput.toIntOrNull()
        while (numNeighbors == null) {
            println("Invalid input. Please enter an integer")
            neighborsInput = readLine()!!
            numNeighbors = neighborsInput.toIntOrNull()
        }

        println(n.test(numNeighbors, slength, swidth, plength, pwidth))

        // Ask if the user wants to continue
        println("Do you want to make another prediction? (yes/no)")
        val response = readLine()!!.toLowerCase()
        continueLoop = response == "yes"
    }
}

